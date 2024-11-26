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

// Constants and Interfaces
const MAX_TWEET_LENGTH = 280;
const REPLY_AGE_LIMIT_MS = 5 * 24 * 60 * 60 * 1000; // 5 days
const CLEANUP_AGE = 7 * 24 * 60 * 60 * 1000; // 7 days
const MIN_RETRY_DELAY = 5 * 60 * 1000; // 5 minutes

interface SpecialInteraction {
    handle: string;
    topics: string[];
    templates: string[];
    probability: number;
}

interface ConversationData {
    timestamp: number;
    replies: string[];
    lastReplyId: string;
}

// Special Interactions Configuration
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

// Utility Functions
function truncateToCompleteSentence(text: string): string {
    if (text.length <= MAX_TWEET_LENGTH) {
        return text;
    }

    const truncatedAtPeriod = text.slice(0, text.lastIndexOf(".", MAX_TWEET_LENGTH) + 1);
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    const truncatedAtSpace = text.slice(0, text.lastIndexOf(" ", MAX_TWEET_LENGTH));
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    return text.slice(0, MAX_TWEET_LENGTH - 3).trim() + "...";
}

// Conversation Cache Manager
class ConversationCache {
    private runtime: IAgentRuntime;
    private username: string;

    constructor(runtime: IAgentRuntime, username: string) {
        this.runtime = runtime;
        this.username = username;
    }

    private getKey(conversationId: string): string {
        return `twitter/${this.username}/conversations/${conversationId}`;
    }

    async isParticipating(conversationId: string): Promise<boolean> {
        try {
            const key = this.getKey(conversationId);
            return !!(await this.runtime.cacheManager.get<ConversationData>(key));
        } catch (error) {
            elizaLogger.error("Error checking conversation participation:", error);
            return false;
        }
    }

    async markParticipating(tweet: Tweet): Promise<void> {
        try {
            const key = this.getKey(tweet.conversationId);
            const data: ConversationData = {
                timestamp: Date.now(),
                replies: [tweet.id],
                lastReplyId: tweet.id
            };
            await this.runtime.cacheManager.set(key, data);
            elizaLogger.debug(`Marked participating in conversation ${tweet.conversationId}`);
        } catch (error) {
            elizaLogger.error("Error marking conversation participation:", error);
        }
    }

    async cleanup(): Promise<void> {
        try {
            const now = Date.now();
            const pattern = `twitter/${this.username}/conversations/*`;
            const keys = await this.runtime.cacheManager.keys(pattern);

            for (const key of keys) {
                const data = await this.runtime.cacheManager.get<ConversationData>(key);
                if (data && (now - data.timestamp > CLEANUP_AGE)) {
                    await this.runtime.cacheManager.delete(key);
                    elizaLogger.debug(`Cleaned up old conversation: ${key}`);
                }
            }
        } catch (error) {
            elizaLogger.error("Error cleaning up conversations:", error);
        }
    }
}

// Main Twitter Post Client
export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    private conversationCache: ConversationCache | null = null;
    private lastInteractionType: string | null = null;
    private isRunning: boolean = false;
    private isInitialized: boolean = false;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    private async initialize() {
        if (!this.isInitialized) {
            if (!this.client.profile) {
                await this.client.init();
            }
            this.conversationCache = new ConversationCache(
                this.runtime,
                this.client.profile.username
            );
            this.isInitialized = true;
        }
    }

    private async ensureInitialized() {
        if (!this.isInitialized) {
            await this.initialize();
        }
        if (!this.conversationCache) {
            throw new Error("Conversation cache not initialized");
        }
        return this.conversationCache;
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
            const cache = await this.ensureInitialized();

            // Skip self-tweets
            if (tweet.username === this.client.profile.username) {
                elizaLogger.debug(`Skipping own tweet ${tweet.id}`);
                return false;
            }

            // Check age
            const tweetDate = new Date(tweet.timestamp).getTime();
            const ageInMs = Date.now() - tweetDate;
            if (ageInMs > REPLY_AGE_LIMIT_MS) {
                elizaLogger.debug(`Tweet ${tweet.id} is too old (${Math.floor(ageInMs / (24 * 60 * 60 * 1000))} days)`);
                return false;
            }

            // Check conversation participation
            const isParticipating = await cache.isParticipating(tweet.conversationId);
            if (isParticipating) {
                elizaLogger.debug(`Already participating in conversation ${tweet.conversationId}`);
                return false;
            }

            // Mark as participating immediately
            await cache.markParticipating(tweet);
            return true;

        } catch (error) {
            elizaLogger.error("Error validating tweet for reply:", error);
            return false;
        }
    }

    async processNewTweet(tweet: Tweet): Promise<void> {
        try {
            const cache = await this.ensureInitialized();

            const isParticipating = await cache.isParticipating(tweet.conversationId);
            if (isParticipating) {
                elizaLogger.debug(`Skipping tweet ${tweet.id} - already in conversation ${tweet.conversationId}`);
                return;
            }

            const isValid = await this.validateTweetForReply(tweet);
            if (!isValid) {
                return;
            }

            elizaLogger.debug(`Processing new tweet ${tweet.id}`);
            // Add your tweet processing logic here
            
        } catch (error) {
            elizaLogger.error("Error processing tweet:", error);
        }
    }

    async start(postImmediately: boolean = false) {
        if (this.isRunning) {
            elizaLogger.warn("Twitter post client is already running");
            return;
        }

        await this.initialize();
        this.isRunning = true;

        const generateNewTweetLoop = async () => {
            try {
                const lastPost = await this.runtime.cacheManager.get<{
                    timestamp: number;
                }>(
                    `twitter/${this.runtime.getSetting("TWITTER_USERNAME")}/lastPost`
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
                    }, MIN_RETRY_DELAY);
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

    private async generateNewTweet() {
        elizaLogger.log("Generating new tweet");

        try {
            await this.ensureInitialized();
            
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
                content = await this.generateRegularTweet();
            }

            content = truncateToCompleteSentence(content);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have posted tweet: ${content}`);
                return;
            }

            const tweet = await this.postTweet(content);
            if (tweet) {
                await this.updateCachesAndMemory(tweet, content);
            }

        } catch (error) {
            elizaLogger.error("Error generating new tweet:", error);
            throw error;
        }
    }

    private async generateRegularTweet(): Promise<string> {
        let homeTimeline: Tweet[] = [];
        try {
            const cachedTimeline = await this.client.getCachedTimeline();
            homeTimeline = cachedTimeline || await this.client.fetchHomeTimeline(10);
            if (!cachedTimeline) {
                await this.client.cacheTimeline(homeTimeline);
            }
        } catch (error) {
            elizaLogger.error("Error fetching timeline:", error);
            homeTimeline = [];
        }

        const formattedTimeline = `# ${this.runtime.character.name}'s Home Timeline\n\n` +
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
                timeline: formattedTimeline,
            }
        );

        const context = composeContext({
            state,
            template: this.runtime.character.templates?.twitterPostTemplate || twitterPostTemplate,
        });

        const newTweetContent = await generateText({
            runtime: this.runtime,
            context,
            modelClass: ModelClass.SMALL,
        });
        private async postTweet(content: string): Promise<Tweet | null> {
        try {
            elizaLogger.log(`Posting new tweet:\n ${content}`);

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.sendTweet(content)
            );
            
            if (!result.ok) {
                throw new Error(`Failed to post tweet: ${result.status} ${result.statusText}`);
            }

            const body = await result.json();
            return this.createTweetObject(body.data.create_tweet.tweet_results.result);

        } catch (error) {
            elizaLogger.error("Error posting tweet:", error);
            return null;
        }
    }

    private createTweetObject(tweetResult: any, parentTweet?: Tweet): Tweet {
        return {
            id: tweetResult.rest_id,
            name: this.client.profile.screenName,
            username: this.client.profile.username,
            text: tweetResult.legacy.full_text,
            conversationId: parentTweet?.conversationId || tweetResult.legacy.conversation_id_str,
            createdAt: tweetResult.legacy.created_at,
            timestamp: new Date(tweetResult.legacy.created_at).getTime(),
            userId: this.client.profile.id,
            inReplyToStatusId: parentTweet?.id || tweetResult.legacy.in_reply_to_status_id_str,
            permanentUrl: `https://twitter.com/${this.runtime.getSetting("TWITTER_USERNAME")}/status/${tweetResult.rest_id}`,
            hashtags: [],
            mentions: [],
            photos: [],
            thread: [],
            urls: [],
            videos: [],
        } as Tweet;
    }

    private async updateCachesAndMemory(tweet: Tweet, content: string): Promise<void> {
        try {
            const cache = await this.ensureInitialized();

            // Update last post cache
            await this.runtime.cacheManager.set(
                `twitter/${this.client.profile.username}/lastPost`,
                {
                    id: tweet.id,
                    timestamp: Date.now(),
                }
            );

            // Cache the tweet
            await this.client.cacheTweet(tweet);

            // Update timeline
            const homeTimeline = await this.client.getCachedTimeline() || [];
            homeTimeline.unshift(tweet);
            await this.client.cacheTimeline(homeTimeline);

            // Mark conversation if it's a reply
            if (tweet.inReplyToStatusId) {
                await cache.markParticipating(tweet);
            }

            // Create memory
            const roomId = stringToUuid(tweet.conversationId + "-" + this.runtime.agentId);
            await this.runtime.ensureRoomExists(roomId);
            await this.runtime.ensureParticipantInRoom(this.runtime.agentId, roomId);

            await this.runtime.messageManager.createMemory({
                id: stringToUuid(tweet.id + "-" + this.runtime.agentId),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                content: {
                    text: content.trim(),
                    url: tweet.permanentUrl,
                    source: "twitter",
                    conversationId: tweet.conversationId,
                    isReply: !!tweet.inReplyToStatusId,
                },
                roomId,
                embedding: embeddingZeroVector,
                createdAt: tweet.timestamp,
            });

            elizaLogger.log(`Tweet posted and cached: ${tweet.permanentUrl}`);
        } catch (error) {
            elizaLogger.error("Error updating caches and memory:", error);
            throw error;
        }
    }

    async reply(tweet: Tweet, content: string): Promise<Tweet | null> {
        try {
            const cache = await this.ensureInitialized();
            
            // Quick check before validation
            const isParticipating = await cache.isParticipating(tweet.conversationId);
            if (isParticipating) {
                elizaLogger.debug(`Skipping reply - already in conversation ${tweet.conversationId}`);
                return null;
            }

            const isValid = await this.validateTweetForReply(tweet);
            if (!isValid) {
                return null;
            }

            content = truncateToCompleteSentence(content);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have replied to tweet ${tweet.id}: ${content}`);
                return null;
            }

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.reply(content, tweet.id)
            );

            if (!result.ok) {
                throw new Error(`Reply failed: ${result.status} ${result.statusText}`);
            }

            const body = await result.json();
            const replyTweet = this.createTweetObject(body.data.create_tweet.tweet_results.result, tweet);

            await this.updateCachesAndMemory(replyTweet, content);
            elizaLogger.log(`Reply posted: ${replyTweet.permanentUrl}`);
            
            return replyTweet;

        } catch (error) {
            elizaLogger.error(`Reply error for tweet ${tweet.id}:`, error);
            return null;
        }
    }
}

// Helper function to clean up old records
export async function cleanupOldRecords(client: TwitterPostClient): Promise<void> {
    try {
        await client.ensureInitialized();
        const cache = await client.ensureInitialized();
        await cache.cleanup();
    } catch (error) {
        elizaLogger.error("Error cleaning up old records:", error);
    }
}

// Helper to initialize the client with cleanup routine
export function initializeTwitterClient(
    client: ClientBase, 
    runtime: IAgentRuntime,
    cleanupInterval: number = 24 * 60 * 60 * 1000
): TwitterPostClient {
    const twitterClient = new TwitterPostClient(client, runtime);

    // Set up periodic cleanup
    setInterval(async () => {
        await cleanupOldRecords(twitterClient);
    }, cleanupInterval);

    elizaLogger.log("Twitter client initialized with conversation control");
    return twitterClient;
}
