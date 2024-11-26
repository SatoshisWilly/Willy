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

// Constants
const MAX_TWEET_LENGTH = 280;
const REPLY_AGE_LIMIT_MS = 5 * 24 * 60 * 60 * 1000; // 5 days
const CLEANUP_AGE = 7 * 24 * 60 * 60 * 1000; // 7 days
const MIN_RETRY_DELAY = 5 * 60 * 1000; // 5 minutes

// Types
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

// Template Configuration
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

class ConversationCache {
    private runtime: IAgentRuntime;
    private username: string;

    constructor(runtime: IAgentRuntime, username: string) {
        this.runtime = runtime;
        this.username = username;
    }

    async isParticipating(conversationId: string): Promise<boolean> {
        try {
            const key = `twitter/${this.username}/conversations/${conversationId}`;
            return !!(await this.runtime.cacheManager.get<ConversationData>(key));
        } catch (error) {
            elizaLogger.error("Error checking conversation participation:", error);
            return false;
        }
    }

    async markParticipating(tweet: Tweet): Promise<void> {
        try {
            const key = `twitter/${this.username}/conversations/${tweet.conversationId}`;
            await this.runtime.cacheManager.set(key, {
                timestamp: Date.now(),
                replies: [tweet.id],
                lastReplyId: tweet.id
            });
            elizaLogger.debug(`Marked participating in conversation ${tweet.conversationId}`);
        } catch (error) {
            elizaLogger.error("Error marking conversation participation:", error);
        }
    }

    async cleanup(): Promise<void> {
        try {
            const pattern = `twitter/${this.username}/conversations/*`;
            const keys = await this.runtime.cacheManager.keys(pattern);
            const now = Date.now();

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

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    private conversationCache: ConversationCache | null = null;
    private lastInteractionType: string | null = null;
    private isRunning = false;
    private isInitialized = false;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    async initialize(): Promise<void> {
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

    private async ensureInitialized(): Promise<ConversationCache> {
        if (!this.isInitialized) {
            await this.initialize();
        }
        if (!this.conversationCache) {
            throw new Error("Conversation cache not initialized");
        }
        return this.conversationCache;
    }

    #shouldGenerateSpecialInteraction = (): string | null => {
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
    };

    #generateSpecialInteraction = (type: string): string => {
        const interaction = SPECIAL_INTERACTIONS[type];
        const templates = interaction.templates;
        const template = templates[Math.floor(Math.random() * templates.length)];
        return `${interaction.handle} ${template}`;
    };

    #validateTweetForReply = async (tweet: Tweet): Promise<boolean> => {
        try {
            const cache = await this.ensureInitialized();

            // Skip self-tweets
            if (tweet.username === this.client.profile.username) {
                elizaLogger.debug(`Skipping own tweet ${tweet.id}`);
                return false;
            }

            // Check tweet age
            const tweetAge = Date.now() - new Date(tweet.timestamp).getTime();
            if (tweetAge > REPLY_AGE_LIMIT_MS) {
                elizaLogger.debug(`Tweet ${tweet.id} is too old (${Math.floor(tweetAge / (24 * 60 * 60 * 1000))} days)`);
                return false;
            }

            // Check if already participating
            const isParticipating = await cache.isParticipating(tweet.conversationId);
            if (isParticipating) {
                elizaLogger.debug(`Already participated in conversation ${tweet.conversationId}`);
                return false;
            }

            // Mark as participating
            await cache.markParticipating(tweet);
            return true;

        } catch (error) {
            elizaLogger.error("Error validating tweet for reply:", error);
            return false;
        }
    };

    #postTweet = async (content: string): Promise<Tweet | null> => {
        try {
            elizaLogger.log(`Posting new tweet:\n ${content}`);

            const result = await this.client.requestQueue.add(
                async () => await this.client.twitterClient.sendTweet(content)
            );
            
            if (!result.ok) {
                throw new Error(`Failed to post tweet: ${result.status} ${result.statusText}`);
            }

            const body = await result.json();
            return this.#createTweetObject(body.data.create_tweet.tweet_results.result);

        } catch (error) {
            elizaLogger.error("Error posting tweet:", error);
            return null;
        }
    };

    #createTweetObject = (tweetResult: any, parentTweet?: Tweet): Tweet => ({
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
    });

    #updateCachesAndMemory = async (tweet: Tweet, content: string): Promise<void> => {
        try {
            const cache = await this.ensureInitialized();

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

            if (tweet.inReplyToStatusId) {
                await cache.markParticipating(tweet);
            }

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
    };

    #formatTimeline = (tweets: Tweet[]): string => {
        return `# ${this.runtime.character.name}'s Home Timeline\n\n` +
            tweets
                .map((tweet) => (
                    `#${tweet.id}\n${tweet.name} (@${tweet.username})${
                        tweet.inReplyToStatusId ? `\nIn reply to: ${tweet.inReplyToStatusId}` : ""
                    }\n${new Date(tweet.timestamp).toDateString()}\n\n${tweet.text}\n---\n`
                ))
                .join("\n");
    };

    #generateTweetContent = async (formattedTimeline: string): Promise<string> => {
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

        return newTweetContent.replaceAll(/\\n/g, "\n").trim();
    };

    async start(postImmediately = false): Promise<void> {
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

    async stop(): Promise<void> {
        this.isRunning = false;
        elizaLogger.log("Twitter post client stopped");
    }

    async processNewTweet(tweet: Tweet): Promise<void> {
        try {
            const cache = await this.ensureInitialized();
            
            // Check if already in conversation
            const isParticipating = await cache.isParticipating(tweet.conversationId);
            if (isParticipating) {
                elizaLogger.debug(`Skipping tweet ${tweet.id} - already in conversation ${tweet.conversationId}`);
                return;
            }

            // Validate tweet
            const isValid = await this.#validateTweetForReply(tweet);
            if (!isValid) {
                return;
            }

            elizaLogger.debug(`Processing new tweet ${tweet.id}`);
            // Add your tweet processing logic here
            
        } catch (error) {
            elizaLogger.error("Error processing tweet:", error);
        }
    }

    async generateNewTweet(): Promise<void> {
        elizaLogger.log("Generating new tweet");

        try {
            const cache = await this.ensureInitialized();
            
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            let content: string;
            const specialInteractionType = this.#shouldGenerateSpecialInteraction();

            if (specialInteractionType) {
                content = this.#generateSpecialInteraction(specialInteractionType);
            } else {
                // Get timeline
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

                const formattedTimeline = this.#formatTimeline(homeTimeline);
                content = await this.#generateTweetContent(formattedTimeline);
            }

            content = truncateToCompleteSentence(content);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(`Dry run: would have posted tweet: ${content}`);
                return;
            }

            const tweet = await this.#postTweet(content);
            if (tweet) {
                await this.#updateCachesAndMemory(tweet, content);
            }

        } catch (error) {
            elizaLogger.error("Error generating new tweet:", error);
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

            const isValid = await this.#validateTweetForReply(tweet);
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
            const replyTweet = this.#createTweetObject(body.data.create_tweet.tweet_results.result, tweet);

            await this.#updateCachesAndMemory(replyTweet, content);
            elizaLogger.log(`Reply posted: ${replyTweet.permanentUrl}`);
            
            return replyTweet;

        } catch (error) {
            elizaLogger.error(`Reply error for tweet ${tweet.id}:`, error);
            return null;
        }
    }
}

// Utility function for text truncation
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

// Helper function to initialize client with cleanup routine
export function initializeTwitterClient(
    client: ClientBase, 
    runtime: IAgentRuntime,
    cleanupInterval = 24 * 60 * 60 * 1000 // Default to daily cleanup
): TwitterPostClient {
    const twitterClient = new TwitterPostClient(client, runtime);

    // Set up periodic cleanup
    setInterval(async () => {
        const cache = await twitterClient.ensureInitialized();
        await cache.cleanup();
    }, cleanupInterval);

    elizaLogger.log("Twitter client initialized with conversation control");
    return twitterClient;
}
