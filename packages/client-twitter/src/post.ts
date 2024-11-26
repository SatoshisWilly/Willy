import { Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateText,
    embeddingZeroVector,
    IAgentRuntime,
    ModelClass,
    stringToUuid,
    parseBooleanFromText,
    elizaLogger,
} from "@ai16z/eliza";
import { ClientBase } from "./base.ts";
import { SpecialInteraction, validateTwitterConfig } from "./environment.ts";
import { sample } from 'lodash';

const twitterPostTemplate = `{{timeline}}

# Knowledge
{{knowledge}}

About {{agentName}} (@{{twitterUserName}}):
{{bio}}
{{lore}}
{{postDirections}}

{{providers}}

{{recentPosts}}

{{characterPostExamples}}

# Task: Generate a post in the voice and style of {{agentName}}, aka @{{twitterUserName}}
Write a single sentence post that is {{adjective}} about {{topic}} (without mentioning {{topic}} directly), from the perspective of {{agentName}}. Try to write something totally different than previous posts. Do not add commentary or acknowledge this request, just write the post.
Your response should not contain any questions. Brief, concise statements only. use emojis. use hashtags #Willy, #WillytoaBilly, #WillytuahBilly, #BeleiveWithUs, #WillysBelievers, #ArmyofBelievers, #NothingPumpsHarder  Use \\n\\n (double spaces) between statements.

const MAX_TWEET_LENGTH = 280;

interface PostSchedule {
    minMinutes: number;
    maxMinutes: number;
    lastPostTime: number;
}

/**
 * Truncate text to fit within the Twitter character limit, ensuring it ends at a complete sentence.
 */
function truncateToCompleteSentence(text: string): string {
    if (text.length <= MAX_TWEET_LENGTH) {
        return text;
    }

    // Attempt to truncate at the last period within the limit
    const truncatedAtPeriod = text.slice(
        0,
        text.lastIndexOf(".", MAX_TWEET_LENGTH) + 1
    );
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    // If no period is found, truncate to the nearest whitespace
    const truncatedAtSpace = text.slice(
        0,
        text.lastIndexOf(" ", MAX_TWEET_LENGTH)
    );
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    // Fallback: Hard truncate and add ellipsis
    return text.slice(0, MAX_TWEET_LENGTH - 3).trim() + "...";
}

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    private schedule: PostSchedule;
    private isPosting: boolean = false;
    private lastSpecialInteraction: Record<string, number> = {};
    private specialInteractions: Record<string, SpecialInteraction> = {};
    private specialInteractionCooldown: number;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
        
        // Initialize schedule with defaults or from runtime settings
        this.schedule = {
            minMinutes: parseInt(runtime.getSetting("POST_INTERVAL_MIN")) || 90,
            maxMinutes: parseInt(runtime.getSetting("POST_INTERVAL_MAX")) || 180,
            lastPostTime: 0
        };

        this.specialInteractionCooldown = 24 * 60 * 60 * 1000; // Default 24h
    }

    private async loadSpecialInteractions() {
        try {
            const config = await validateTwitterConfig(this.runtime);
            this.specialInteractions = config.TWITTER_SPECIAL_INTERACTIONS;
            this.specialInteractionCooldown = config.TWITTER_SPECIAL_INTERACTION_COOLDOWN;

            // Initialize tracking for all configured interactions
            Object.keys(this.specialInteractions).forEach(key => {
                this.lastSpecialInteraction[key] = 0;
            });

            elizaLogger.log(`Loaded ${Object.keys(this.specialInteractions).length} special interactions`);
            elizaLogger.debug("Special interactions configured:", 
                Object.keys(this.specialInteractions));
        } catch (error) {
            elizaLogger.error("Error loading special interactions:", error);
            this.specialInteractions = {};
        }
    }

    async start(postImmediately: boolean = false) {
        if (!this.client.profile) {
            await this.client.init();
        }

        // Load special interactions
        await this.loadSpecialInteractions();

        // Load last post time from cache
        const lastPost = await this.runtime.cacheManager.get<{
            timestamp: number;
        }>(
            `twitter/${this.runtime.getSetting("TWITTER_USERNAME")}/lastPost`
        );
        this.schedule.lastPostTime = lastPost?.timestamp ?? 0;

        const generateNewTweetLoop = async () => {
            try {
                const now = Date.now();
                const timeSinceLastPost = now - this.schedule.lastPostTime;
                const randomMinutes = Math.floor(
                    Math.random() * 
                    (this.schedule.maxMinutes - this.schedule.minMinutes + 1)
                ) + this.schedule.minMinutes;
                const delay = randomMinutes * 60 * 1000;

                if (timeSinceLastPost >= delay && !this.isPosting) {
                    await this.generateNewTweet();
                }

                // Schedule next tweet
                setTimeout(() => generateNewTweetLoop(), delay);

                elizaLogger.log(`Next tweet scheduled in ${randomMinutes} minutes`);
            } catch (error) {
                elizaLogger.error("Error in tweet generation loop:", error);
                // Retry after delay on error
                setTimeout(() => generateNewTweetLoop(), 5 * 60 * 1000);
            }
        };

        if (
            this.runtime.getSetting("POST_IMMEDIATELY") != null &&
            this.runtime.getSetting("POST_IMMEDIATELY") !== ""
        ) {
            postImmediately = parseBooleanFromText(
                this.runtime.getSetting("POST_IMMEDIATELY")
            );
        }

        if (postImmediately) {
            await this.generateNewTweet();
        }

        generateNewTweetLoop();
    }

    private shouldTriggerSpecialInteraction(type: string): boolean {
        const now = Date.now();
        const lastTime = this.lastSpecialInteraction[type] || 0;
        const interaction = this.specialInteractions[type];

        if (!interaction) return false;

        // Check cooldown
        if (now - lastTime < this.specialInteractionCooldown) {
            elizaLogger.debug(`Special interaction ${type} on cooldown for ${Math.floor((this.specialInteractionCooldown - (now - lastTime)) / 1000 / 60)} more minutes`);
            return false;
        }

        // Random chance based on probability
        const shouldTrigger = Math.random() < interaction.probability;
        if (shouldTrigger) {
            elizaLogger.log(`Triggering special interaction: ${type}`);
        }
        return shouldTrigger;
    }

    private async generateSpecialTweet(type: string): Promise<string | null> {
        const interaction = this.specialInteractions[type];
        if (!interaction) return null;

        // Get a random template and topic
        const template = sample(interaction.templates);
        const topic = sample(interaction.topics);

        if (!template || !topic) {
            elizaLogger.warn(`Missing template or topic for special interaction ${type}`);
            return null;
        }

        // Create tweet with mention
        const tweet = `${interaction.handle} ${template}`;

        // Update last interaction time
        this.lastSpecialInteraction[type] = Date.now();

        elizaLogger.log(`Generated special tweet for ${type}: ${tweet}`);
        return tweet;
    }

    private async generateNewTweet() {
        if (this.isPosting) {
            elizaLogger.log("Tweet generation already in progress, skipping");
            return;
        }

        this.isPosting = true;
        elizaLogger.log("Generating new tweet");

        try {
            // Check for special interactions first
            let content: string | null = null;
            
            // Randomly select one special interaction type to check
            const interactionTypes = Object.keys(this.specialInteractions);
            const randomType = sample(interactionTypes);
            
            if (randomType && this.shouldTriggerSpecialInteraction(randomType)) {
                content = await this.generateSpecialTweet(randomType);
            }

            // If no special interaction, generate normal tweet
            if (!content) {
                await this.runtime.ensureUserExists(
                    this.runtime.agentId,
                    this.client.profile.username,
                    this.runtime.character.name,
                    "twitter"
                );

                let homeTimeline: Tweet[] = [];
                const cachedTimeline = await this.client.getCachedTimeline();

                if (cachedTimeline) {
                    homeTimeline = cachedTimeline;
                } else {
                    homeTimeline = await this.client.fetchHomeTimeline(10);
                    await this.client.cacheTimeline(homeTimeline);
                }

                const formattedHomeTimeline =
                    `# ${this.runtime.character.name}'s Home Timeline\n\n` +
                    homeTimeline
                        .map((tweet) => {
                            return `#${tweet.id}\n${tweet.name} (@${tweet.username})${
                                tweet.inReplyToStatusId
                                    ? `\nIn reply to: ${tweet.inReplyToStatusId}`
                                    : ""
                            }\n${new Date(tweet.timestamp * 1000).toDateString()}\n\n${
                                tweet.text
                            }\n---\n`;
                        })
                        .join("\n");

                const topics = this.runtime.character.topics.join(", ");

                const state = await this.runtime.composeState(
                    {
                        userId: this.runtime.agentId,
                        roomId: stringToUuid("twitter_generate_room"),
                        agentId: this.runtime.agentId,
                        content: {
                            text: topics,
                            action: "",
                        },
                    },
                    {
                        twitterUserName: this.client.profile.username,
                        timeline: formattedHomeTimeline,
                    }
                );

                const context = composeContext({
                    state,
                    template:
                        this.runtime.character.templates?.twitterPostTemplate ||
                        twitterPostTemplate,
                });

                const newTweetContent = await generateText({
                    runtime: this.runtime,
                    context,
                    modelClass: ModelClass.SMALL,
                });

                // Replace \n with proper line breaks and trim excess spaces
                const formattedTweet = newTweetContent
                    .replaceAll(/\\n/g, "\n")
                    .trim();

                content = truncateToCompleteSentence(formattedTweet);
            }

            if (!content) {
                elizaLogger.warn("No content generated for tweet");
                return;
            }

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have posted tweet: ${content}`);
                return;
            }

            elizaLogger.log(`Posting new tweet:\n ${content}`);

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.sendTweet(content)
            );
            const body = await result.json();
            const tweetResult = body.data.create_tweet.tweet_results.result;

            const tweet: Tweet = {
                id: tweetResult.rest_id,
                name: this.client.profile.screenName,
                username: this.client.profile.username,
                text: tweetResult.legacy.full_text,
                conversationId: tweetResult.legacy.conversation_id_str,
                timestamp: new Date(tweetResult.legacy.created_at).getTime() / 1000,
                userId: this.client.profile.id,
                inReplyToStatusId: tweetResult.legacy.in_reply_to_status_id_str,
                permanentUrl: `https://twitter.com/${this.runtime.getSetting("TWITTER_USERNAME")}/status/${tweetResult.rest_id}`,
                hashtags: tweetResult.legacy.entities?.hashtags || [],
                mentions: tweetResult.legacy.entities?.user_mentions || [],
                photos: [],
                thread: [],
                urls: tweetResult.legacy.entities?.urls || [],
                videos: [],
            };

            // Update timeline and caches
            let homeTimeline = await this.client.getCachedTimeline() || [];
            homeTimeline.unshift(tweet);
            await this.client.cacheTimeline(homeTimeline);
            await this.client.cacheTweet(tweet);

            // Update last post timestamp
            const postInfo = {
                id: tweet.id,
                timestamp: Date.now(),
            };
            await this.runtime.cacheManager.set(
                `twitter/${this.client.profile.username}/lastPost`,
                postInfo
            );
            this.schedule.lastPostTime = postInfo.timestamp;

            elizaLogger.log(`Tweet posted:\n ${tweet.permanentUrl}`);

            // Create room and memory entries
            const roomId = stringToUuid(
                tweet.conversationId + "-" + this.runtime.agentId
            );

            await this.runtime.ensureRoomExists(roomId);
            await this.runtime.ensureParticipantInRoom(
                this.runtime.agentId,
                roomId
            );

            await this.runtime.messageManager.createMemory({
                id: stringToUuid(tweet.id + "-" + this.runtime.agentId),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                content: {
                    text: content.trim(),
                    url: tweet.permanentUrl,
                    source: "twitter",
                },
                roomId,
                embedding: embeddingZeroVector,
                createdAt: tweet.timestamp * 1000,
            });

        } catch (error) {
            elizaLogger.error("Error generating or posting tweet:", error);
        } finally {
            this.isPosting = false;
        }
    }
}
