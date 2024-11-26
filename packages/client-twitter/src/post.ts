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
import { ClientBase } from "./base";
import type { SpecialInteraction } from "./enviroment.ts";
import { validateTwitterConfig } from "./enviroment.ts";
import { sample } from 'lodash';

const MAX_TWEET_LENGTH = 280;

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
Your response should not contain any questions. Brief, concise statements only. use emojis. use hashtags #Willy, #WillytoaBilly, #WillytuahBilly, #BeleiveWithUs, #WillysBelievers, #ArmyofBelievers, #NothingPumpsHarder  Use \\n\\n (double spaces) between statements.`;

interface PostSchedule {
    minMinutes: number;
    maxMinutes: number;
    lastPostTime: number;
}

interface PostResponse {
    content: string;
    isSpecialInteraction: boolean;
    interactionType?: string;
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
        
        this.schedule = {
            minMinutes: parseInt(runtime.getSetting("POST_INTERVAL_MIN")) || 90,
            maxMinutes: parseInt(runtime.getSetting("POST_INTERVAL_MAX")) || 180,
            lastPostTime: 0
        };

        this.specialInteractionCooldown = 24 * 60 * 60 * 1000;
    }

    private async loadSpecialInteractions(): Promise<void> {
        try {
            const config = await validateTwitterConfig(this.runtime);
            this.specialInteractions = config.TWITTER_SPECIAL_INTERACTIONS;
            this.specialInteractionCooldown = config.TWITTER_SPECIAL_INTERACTION_COOLDOWN;

            Object.keys(this.specialInteractions).forEach(key => {
                this.lastSpecialInteraction[key] = 0;
            });

            elizaLogger.log("Loaded special interactions:", Object.keys(this.specialInteractions).length);
            elizaLogger.debug("Configured interactions:", Object.keys(this.specialInteractions));
        } catch (error) {
            elizaLogger.error("Error loading special interactions:", error);
            this.specialInteractions = {};
        }
    }

    private shouldTriggerSpecialInteraction(type: string): boolean {
        const now = Date.now();
        const lastTime = this.lastSpecialInteraction[type] || 0;
        const interaction = this.specialInteractions[type];

        if (!interaction) {
            return false;
        }

        if (now - lastTime < this.specialInteractionCooldown) {
            elizaLogger.debug(`Special interaction ${type} on cooldown: ${Math.floor((this.specialInteractionCooldown - (now - lastTime)) / 1000 / 60)}m remaining`);
            return false;
        }

        const shouldTrigger = Math.random() < interaction.probability;
        if (shouldTrigger) {
            elizaLogger.log(`Triggering special interaction: ${type}`);
        }
        return shouldTrigger;
    }

    private async generateSpecialTweet(type: string): Promise<PostResponse | null> {
        const interaction = this.specialInteractions[type];
        if (!interaction) {
            return null;
        }

        const template = sample(interaction.templates);
        const topic = sample(interaction.topics);

        if (!template || !topic) {
            elizaLogger.warn(`Missing template or topic for interaction ${type}`);
            return null;
        }

        const content = `${interaction.handle} ${template}`;
        this.lastSpecialInteraction[type] = Date.now();

        elizaLogger.log(`Generated special tweet for ${type}: ${content}`);
        return {
            content,
            isSpecialInteraction: true,
            interactionType: type
        };
    }

    async start(postImmediately: boolean = false): Promise<void> {
        if (!this.client.profile) {
            await this.client.init();
        }

        await this.loadSpecialInteractions();

        const lastPost = await this.runtime.cacheManager.get<{
            timestamp: number;
        }>(
            `twitter/${this.runtime.getSetting("TWITTER_USERNAME")}/lastPost`
        );
        this.schedule.lastPostTime = lastPost?.timestamp ?? 0;

        const generateNewTweetLoop = async (): Promise<void> => {
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

                setTimeout(() => generateNewTweetLoop(), delay);
                elizaLogger.log(`Next tweet scheduled in ${randomMinutes} minutes`);
            } catch (error) {
                elizaLogger.error("Error in tweet generation loop:", error);
                setTimeout(() => generateNewTweetLoop(), 5 * 60 * 1000);
            }
        };

        const shouldPostImmediately = this.runtime.getSetting("POST_IMMEDIATELY");
        if (shouldPostImmediately != null && shouldPostImmediately !== "") {
            postImmediately = parseBooleanFromText(shouldPostImmediately);
        }

        if (postImmediately) {
            await this.generateNewTweet();
        }

        generateNewTweetLoop();
    }

    private async generateTweetContent(): Promise<PostResponse> {
        // Check for special interactions first
        const interactionTypes = Object.keys(this.specialInteractions);
        const randomType = sample(interactionTypes);
        
        if (randomType && this.shouldTriggerSpecialInteraction(randomType)) {
            const specialTweet = await this.generateSpecialTweet(randomType);
            if (specialTweet) {
                return specialTweet;
            }
        }

        // Generate normal tweet if no special interaction
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

        const formattedTweet = newTweetContent
            .replaceAll(/\\n/g, "\n")
            .trim();

        return {
            content: truncateToCompleteSentence(formattedTweet),
            isSpecialInteraction: false
        };
    }

    private async generateNewTweet(): Promise<void> {
        if (this.isPosting) {
            elizaLogger.log("Tweet generation already in progress, skipping");
            return;
        }

        this.isPosting = true;
        elizaLogger.log("Generating new tweet");

        try {
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            const tweetResponse = await this.generateTweetContent();
            if (!tweetResponse.content) {
                elizaLogger.warn("No content generated for tweet");
                return;
            }

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have posted tweet: ${tweetResponse.content}`);
                return;
            }

            elizaLogger.log(`Posting new tweet:\n ${tweetResponse.content}`);

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.sendTweet(tweetResponse.content)
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
                type: tweetResponse.isSpecialInteraction ? tweetResponse.interactionType : 'normal'
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
                    text: tweetResponse.content.trim(),
                    url: tweet.permanentUrl,
                    source: "twitter",
                    metadata: {
                        isSpecialInteraction: tweetResponse.isSpecialInteraction,
                        interactionType: tweetResponse.interactionType
                    }
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

export default TwitterPostClient;
