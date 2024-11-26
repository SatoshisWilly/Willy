import { Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateText,
    embeddingZeroVector,
    IAgentRuntime,
    ModelClass,
    stringToUuid,
} from "@ai16z/eliza";
import { elizaLogger } from "@ai16z/eliza";
import { ClientBase } from "./base.ts";

interface SpecialInteraction {
    handle: string;
    topics: string[];
    templates: string[];
    probability: number;
}

const SPECIAL_INTERACTIONS: Record<string, SpecialInteraction> = {
    tate: {
        handle: "@Cobratate",
        topics: ["sold 2M Willy", "paper hands", "missing out", "future regret"],
        templates: [
            "Imagine selling your community-gifted 2M Willy... some people just don't see the vision",
            "When Willy hits a billion, wonder how that 2M sale will feel",
            "Paper hands never make generational wealth... just saying",
            "Some influencers talk diamond hands but fold faster than a lawn chair"
        ],
        probability: 0.15
    },
    hailey: {
        handle: "@HaileyWelchX",
        topics: ["partnership", "Willy tuah Billy", "moon mission", "collaboration"],
        templates: [
            "Let's take this Willy tuah Billy together",
            "Two visionaries, one mission - Willy tuah Billy",
            "When are we partnering to make history?",
            "Your energy + my vision = Willy tuah Billy"
        ],
        probability: 0.15
    },
    dolos: {
        handle: "@dolos_diary",
        topics: ["small willy", "tiny holdings", "weak position"],
        templates: [
            "Heard someone's holding a microscopic bag... must be rough",
            "Some people compensate for their tiny Willy with big talk",
            "Not everyone can handle a big Willy, clearly",
            "Size matters in this game, and someone's coming up short",
            "While others play with their mini Willies, we're building generational wealth"
        ],
        probability: 0.15
    }
};

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

Special Interactions:
1. Occasionally engage with @Cobratate about community-gifted Willy sales and future regret
2. Occasionally flirt with @HaileyWelchX about partnering for 'Willy tuah Billy'`;

const MAX_TWEET_LENGTH = 280;
const REPLY_AGE_LIMIT_MS = 5 * 24 * 60 * 60 * 1000; // 5 days in milliseconds

function truncateToCompleteSentence(text: string): string {
    if (text.length <= MAX_TWEET_LENGTH) {
        return text;
    }

    const truncatedAtPeriod = text.slice(
        0,
        text.lastIndexOf(".", MAX_TWEET_LENGTH) + 1
    );
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    const truncatedAtSpace = text.slice(
        0,
        text.lastIndexOf(" ", MAX_TWEET_LENGTH)
    );
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    return text.slice(0, MAX_TWEET_LENGTH - 3).trim() + "...";
}

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    private lastInteractionType: string | null = null;
    private isRunning: boolean = false;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    private shouldGenerateSpecialInteraction(): string | null {
        if (this.lastInteractionType) {
            this.lastInteractionType = null;
            return null;
        }

        const random = Math.random();
        let cumulativeProbability = 0;

        for (const [type, interaction] of Object.entries(SPECIAL_INTERACTIONS)) {
            cumulativeProbability += interaction.probability;
            if (random < cumulativeProbability) {
                this.lastInteractionType = type;
                return type;
            }
        }

        return null;
    }

    private generateSpecialInteraction(type: string): string {
        const interaction = SPECIAL_INTERACTIONS[type];
        const template = interaction.templates[Math.floor(Math.random() * interaction.templates.length)];
        return `${interaction.handle} ${template}`;
    }

private async validateTweetForReply(tweet: Tweet): Promise<boolean> {
    try {
        // Check tweet age
        const tweetDate = new Date(tweet.timestamp).getTime();
        const ageInMs = Date.now() - tweetDate;
        
        if (ageInMs > REPLY_AGE_LIMIT_MS) {
            elizaLogger.debug(`Tweet ${tweet.id} is too old (${Math.floor(ageInMs / (24 * 60 * 60 * 1000))} days)`);
            return false;
        }

        // Check if we've already replied in this conversation
        const threadKey = `twitter/${this.client.profile.username}/threads/${tweet.conversationId}`;
        const threadData = await this.runtime.cacheManager.get<{
            replies: string[];
            lastReplyTimestamp: number;
        }>(threadKey);

        // If we have any replies in this conversation, don't reply again
        if (threadData && threadData.replies && threadData.replies.length > 0) {
            elizaLogger.debug(`Already replied in thread ${tweet.conversationId} (${threadData.replies.length} replies)`);
            return false;
        }

        return true;
    } catch (error) {
        elizaLogger.error("Error validating tweet for reply:", error);
        return false;
    }
}
private async trackReplyInThread(tweet: Tweet): Promise<void> {
    try {
        const threadKey = `twitter/${this.client.profile.username}/threads/${tweet.conversationId}`;
        
        // Get existing thread data if any
        const existingData = await this.runtime.cacheManager.get<{
            replies: string[];
            lastReplyTimestamp: number;
        }>(threadKey);

        // Create or update thread data
        const newData = {
            replies: existingData ? [...existingData.replies, tweet.id] : [tweet.id],
            lastReplyTimestamp: Date.now()
        };

        await this.runtime.cacheManager.set(threadKey, newData);
        elizaLogger.debug(`Tracked reply ${tweet.id} in thread ${tweet.conversationId}`);
    } catch (error) {
        elizaLogger.error("Error tracking reply in thread:", error);
    }
}
    async start(postImmediately: boolean = false) {
        if (this.isRunning) {
            elizaLogger.warn("Twitter post client is already running");
            return;
        }

        this.isRunning = true;

        if (!this.client.profile) {
            await this.client.init();
        }

        const generateNewTweetLoop = async () => {
            try {
                const lastPost = await this.runtime.cacheManager.get<{
                    timestamp: number;
                }>(
                    "twitter/" +
                    this.runtime.getSetting("TWITTER_USERNAME") +
                    "/lastPost"
                );

                const lastPostTimestamp = lastPost?.timestamp ?? 0;
                const minMinutes = parseInt(this.runtime.getSetting("POST_INTERVAL_MIN")) || 90;
                const maxMinutes = parseInt(this.runtime.getSetting("POST_INTERVAL_MAX")) || 180;
                const randomMinutes = Math.floor(Math.random() * (maxMinutes - minMinutes + 1)) + minMinutes;
                const delay = randomMinutes * 60 * 1000;

                if (Date.now() > lastPostTimestamp + delay) {
                    await this.generateNewTweet();
                }

                if (this.isRunning) {
                    setTimeout(() => {
                        generateNewTweetLoop();
                    }, delay);
                    elizaLogger.log(`Next tweet scheduled in ${randomMinutes} minutes`);
                }
            } catch (error) {
                elizaLogger.error("Error in tweet loop:", error);
                if (this.isRunning) {
                    setTimeout(() => {
                        generateNewTweetLoop();
                    }, 5 * 60 * 1000); // Retry after 5 minutes
                }
            }
        };

        if (postImmediately) {
            await this.generateNewTweet();
        }

        generateNewTweetLoop();
    }

    async stop() {
        this.isRunning = false;
        elizaLogger.log("Twitter post client stopped");
    }

    async processNewTweet(tweet: Tweet): Promise<void> {
        try {
            // Skip if tweet is from self
            if (tweet.username === this.client.profile.username) {
                return;
            }

            // Validate tweet for reply
            const isValid = await this.validateTweetForReply(tweet);
            if (!isValid) {
                return;
            }

            // Process the tweet and generate reply logic here
            // Your existing reply generation logic goes here
            
        } catch (error) {
            elizaLogger.error("Error processing new tweet:", error);
        }
    }

    private async generateNewTweet() {
        elizaLogger.log("Generating new tweet");

        try {
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            let content: string;
            const specialInteractionType = this.shouldGenerateSpecialInteraction();

            if (specialInteractionType) {
                content = this.generateSpecialInteraction(specialInteractionType);
            } else {
                let homeTimeline: Tweet[] = [];
                try {
                    const cachedTimeline = await this.client.getCachedTimeline();
                    if (cachedTimeline) {
                        homeTimeline = cachedTimeline;
                    } else {
                        homeTimeline = await this.client.fetchHomeTimeline(10);
                        await this.client.cacheTimeline(homeTimeline);
                    }
                } catch (error) {
                    elizaLogger.error("Error fetching timeline:", error);
                    homeTimeline = [];
                }

                const formattedHomeTimeline =
                    `# ${this.runtime.character.name}'s Home Timeline\n\n` +
                    homeTimeline
                        .map((tweet) => {
                            return `#${tweet.id}\n${tweet.name} (@${tweet.username})${tweet.inReplyToStatusId ? `\nIn reply to: ${tweet.inReplyToStatusId}` : ""}\n${new Date(tweet.timestamp).toDateString()}\n\n${tweet.text}\n---\n`;
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
                            action: "post",
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

                elizaLogger.debug("generate post prompt:\n" + context);

                const newTweetContent = await generateText({
                    runtime: this.runtime,
                    context,
                    modelClass: ModelClass.SMALL,
                });

                content = newTweetContent
                    .replaceAll(/\\n/g, "\n")
                    .trim();
            }

            content = truncateToCompleteSentence(content);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have posted tweet: ${content}`);
                return;
            }

            try {
                elizaLogger.log(`Posting new tweet:\n ${content}`);

                const result = await this.client.requestQueue.add(
                    async () => await this.client.twitterClient.sendTweet(content)
                );
                
                if (!result.ok) {
                    throw new Error(`Failed to post tweet: ${result.status} ${result.statusText}`);
                }

                const body = await result.json();
                const tweetResult = body.data.create_tweet.tweet_results.result;

                const tweet = {
                    id: tweetResult.rest_id,
                    name: this.client.profile.screenName,
                    username: this.client.profile.username,
                    text: tweetResult.legacy.full_text,
                    conversationId: tweetResult.legacy.conversation_id_str,
                    createdAt: tweetResult.legacy.created_at,
                    userId: this.client.profile.id,
                    inReplyToStatusId: tweetResult.legacy.in_reply_to_status_id_str,
                    permanentUrl: `https://twitter.com/${this.runtime.getSetting("TWITTER_USERNAME")}/status/${tweetResult.rest_id}`,
                    hashtags: [],
                    mentions: [],
                    photos: [],
                    thread: [],
                    urls: [],
                    videos: [],
                } as Tweet;

                await this.runtime.cacheManager.set(
                    `twitter/${this.client.profile.username}/lastPost`,
                    {
                        id: tweet.id,
                        timestamp: Date.now(),
                    }
                );

                await this.client.cacheTweet(tweet);

                const homeTimeline = await this.client.getCachedTimeline() || [];
                homeTimeline.unshift(tweet);
                await this.client.cacheTimeline(homeTimeline);
                elizaLogger.log(`Tweet posted:\n ${tweet.permanentUrl}`);

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
createdAt: new Date(tweet.createdAt).getTime(),
                });
            } catch (error) {
                elizaLogger.error("Error sending tweet:", error);
                throw error;
            }
        } catch (error) {
            elizaLogger.error("Error generating new tweet:", error);
            throw error;
        }
    }

    async reply(tweet: Tweet, content: string): Promise<Tweet | null> {
        try {
            // Validate before replying
            const isValid = await this.validateTweetForReply(tweet);
            if (!isValid) {
                elizaLogger.debug(`Skipping reply to tweet ${tweet.id} - validation failed`);
                return null;
            }

            content = truncateToCompleteSentence(content);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have replied to tweet ${tweet.id}: ${content}`);
                return null;
            }

            elizaLogger.log(`Replying to tweet ${tweet.id}:\n ${content}`);

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.reply(content, tweet.id)
            );

            if (!result.ok) {
                throw new Error(`Failed to post reply: ${result.status} ${result.statusText}`);
            }

            const body = await result.json();
            const tweetResult = body.data.create_tweet.tweet_results.result;

            const replyTweet = {
                id: tweetResult.rest_id,
                name: this.client.profile.screenName,
                username: this.client.profile.username,
                text: tweetResult.legacy.full_text,
                conversationId: tweet.conversationId,
                timestamp: new Date(tweetResult.legacy.created_at).getTime(),
                userId: this.client.profile.id,
                inReplyToStatusId: tweet.id,
                permanentUrl: `https://twitter.com/${this.runtime.getSetting("TWITTER_USERNAME")}/status/${tweetResult.rest_id}`,
                hashtags: [],
                mentions: [],
                photos: [],
                thread: [],
                urls: [],
                videos: [],
            } as Tweet;

            // Track the reply in thread
            await this.trackReplyInThread(replyTweet);

            // Cache the reply tweet
            await this.client.cacheTweet(replyTweet);

            // Update timeline cache
            const homeTimeline = await this.client.getCachedTimeline() || [];
            homeTimeline.unshift(replyTweet);
            await this.client.cacheTimeline(homeTimeline);

            // Create memory of the reply
            const roomId = stringToUuid(tweet.conversationId + "-" + this.runtime.agentId);
            await this.runtime.ensureRoomExists(roomId);
            await this.runtime.ensureParticipantInRoom(this.runtime.agentId, roomId);

            await this.runtime.messageManager.createMemory({
                id: stringToUuid(replyTweet.id + "-" + this.runtime.agentId),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                content: {
                    text: content.trim(),
                    url: replyTweet.permanentUrl,
                    source: "twitter",
                    inReplyTo: tweet.id,
                },
                roomId,
                embedding: embeddingZeroVector,
                createdAt: replyTweet.timestamp,
            });

            elizaLogger.log(`Reply posted: ${replyTweet.permanentUrl}`);
            return replyTweet;

        } catch (error) {
            elizaLogger.error(`Error replying to tweet ${tweet.id}:`, error);
            return null;
        }
    }
}

// Utility function to clean up old thread records
export async function cleanupOldThreadRecords(client: TwitterPostClient): Promise<void> {
    try {
        const threadKeys = await client.runtime.cacheManager.keys(`twitter/${client.client.profile.username}/threads/*`);
        const now = Date.now();
        const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days

        for (const key of threadKeys) {
            const threadData = await client.runtime.cacheManager.get<{
                replies: string[];
                lastReplyTimestamp: number;
            }>(key);
            if (threadData && (now - threadData.lastReplyTimestamp > maxAge)) {
                await client.runtime.cacheManager.delete(key);
                elizaLogger.debug(`Cleaned up old thread record: ${key}`);
            }
        }
    } catch (error) {
        elizaLogger.error("Error cleaning up old thread records:", error);
    }
}

// Helper to initialize the client with cleanup routine
export async function initializeTwitterClient(
    client: ClientBase, 
    runtime: IAgentRuntime,
    cleanupInterval: number = 24 * 60 * 60 * 1000 // Default to daily cleanup
): Promise<TwitterPostClient> {
    const twitterClient = new TwitterPostClient(client, runtime);

    // Set up periodic cleanup
    setInterval(async () => {
        await cleanupOldThreadRecords(twitterClient);
    }, cleanupInterval);

    return twitterClient;
}
