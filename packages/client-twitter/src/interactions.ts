import { SearchMode, Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateMessageResponse,
    generateShouldRespond,
    messageCompletionFooter,
    shouldRespondFooter,
    Content,
    HandlerCallback,
    IAgentRuntime,
    Memory,
    ModelClass,
    State,
    stringToUuid,
    elizaLogger,
} from "@ai16z/eliza";
import { ClientBase } from "./base";
import { buildConversationThread, sendTweet, wait } from "./utils.ts";
import { embeddingZeroVector } from "@ai16z/eliza";

export const twitterMessageHandlerTemplate =
    `{{timeline}}

# Knowledge
{{knowledge}}

# Task: Generate a post for the character {{agentName}}.
About {{agentName}} (@{{twitterUserName}}):
{{bio}}
{{lore}}
{{topics}}

{{providers}}

{{characterPostExamples}}

{{postDirections}}

Recent interactions between {{agentName}} and other users:
{{recentPostInteractions}}

{{recentPosts}}

# Task: Generate a post/reply in the voice, style and perspective of {{agentName}} (@{{twitterUserName}}) while using the thread of tweets as additional context:
Current Post:
{{currentPost}}
Thread of Tweets You Are Replying To:

{{formattedConversation}}

{{actions}}

# Task: Generate a post in the voice, style and perspective of {{agentName}} (@{{twitterUserName}}). Include an action, if appropriate. {{actionNames}}:
{{currentPost}}
` + messageCompletionFooter;

export const twitterShouldRespondTemplate =
    `# INSTRUCTIONS: Determine if {{agentName}} (@{{twitterUserName}}) should respond to the message and participate in the conversation. Do not comment. Just respond with "true" or "false".

Response options are RESPOND, IGNORE and STOP.

{{agentName}} should respond to messages that are directed at them, or participate in conversations that are interesting or relevant to their background, IGNORE messages that are irrelevant to them, and should STOP if the conversation is concluded.

{{agentName}} is in a room with other users and wants to be conversational, but not annoying.
{{agentName}} should RESPOND to messages that are directed at them, or participate in conversations that are interesting or relevant to their background.
If a message is not interesting or relevant, {{agentName}} should IGNORE.
Unless directly RESPONDing to a user, {{agentName}} should IGNORE messages that are very short or do not contain much information.
If a user asks {{agentName}} to stop talking, {{agentName}} should STOP.
If {{agentName}} concludes a conversation and isn't part of the conversation anymore, {{agentName}} should STOP.

{{recentPosts}}

IMPORTANT: {{agentName}} (aka @{{twitterUserName}}) is particularly sensitive about being annoying, so if there is any doubt, it is better to IGNORE than to RESPOND.

{{currentPost}}

Thread of Tweets You Are Replying To:

{{formattedConversation}}

# INSTRUCTIONS: Respond with [RESPOND] if {{agentName}} should respond, or [IGNORE] if {{agentName}} should not respond to the last message and [STOP] if {{agentName}} should stop participating in the conversation.
` + shouldRespondFooter;

// Configuration interface for interaction limits
interface InteractionLimits {
    maxRepliesPerThread: number;      // Maximum replies in a single thread
    maxRepliesPerUser: number;        // Maximum replies to the same user in 24h
    minTimeBetweenReplies: number;    // Minimum time (ms) between replies
    checkIntervalMinutes: number;     // How often to check for new interactions
    replyProbability: number;         // Probability of replying to any given tweet (0-1)
}

export class TwitterInteractionClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    private recentReplies: Map<string, number> = new Map(); // userId -> reply count
    private threadReplies: Map<string, number> = new Map(); // conversationId -> reply count
    private lastReplyTime: Map<string, number> = new Map(); // userId -> timestamp
    private limits: InteractionLimits;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
        
        // Initialize limits with defaults or from runtime settings
        this.limits = {
            maxRepliesPerThread: parseInt(runtime.getSetting("MAX_REPLIES_PER_THREAD")) || 2,
            maxRepliesPerUser: parseInt(runtime.getSetting("MAX_REPLIES_PER_USER")) || 5,
            minTimeBetweenReplies: parseInt(runtime.getSetting("MIN_TIME_BETWEEN_REPLIES")) || 300000, // 5 minutes
            checkIntervalMinutes: parseInt(runtime.getSetting("CHECK_INTERVAL_MINUTES")) || 3,
            replyProbability: parseFloat(runtime.getSetting("REPLY_PROBABILITY")) || 0.5
        };

        // Clean up tracking maps every 24 hours
        setInterval(() => this.cleanupTrackingMaps(), 24 * 60 * 60 * 1000);
    }

    async start() {
        const handleTwitterInteractionsLoop = () => {
            this.handleTwitterInteractions();
            setTimeout(
                handleTwitterInteractionsLoop,
                this.limits.checkIntervalMinutes * 60 * 1000
            );
        };
        handleTwitterInteractionsLoop();
    }

    private cleanupTrackingMaps() {
        const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
        for (const [userId, timestamp] of this.lastReplyTime) {
            if (timestamp < oneDayAgo) {
                this.lastReplyTime.delete(userId);
                this.recentReplies.delete(userId);
            }
        }
        this.threadReplies.clear(); // Reset thread counts daily
    }

    private canReplyToUser(userId: string): boolean {
        const replyCount = this.recentReplies.get(userId) || 0;
        const lastReplyTime = this.lastReplyTime.get(userId) || 0;
        const timeSinceLastReply = Date.now() - lastReplyTime;

        return (
            replyCount < this.limits.maxRepliesPerUser &&
            timeSinceLastReply >= this.limits.minTimeBetweenReplies
        );
    }

    private canReplyInThread(conversationId: string): boolean {
        const threadReplyCount = this.threadReplies.get(conversationId) || 0;
        return threadReplyCount < this.limits.maxRepliesPerThread;
    }

    private updateInteractionTracking(userId: string, conversationId: string) {
        // Update user reply count
        const currentReplies = this.recentReplies.get(userId) || 0;
        this.recentReplies.set(userId, currentReplies + 1);
        
        // Update thread reply count
        const currentThreadReplies = this.threadReplies.get(conversationId) || 0;
        this.threadReplies.set(conversationId, currentThreadReplies + 1);
        
        // Update last reply timestamp
        this.lastReplyTime.set(userId, Date.now());
    }

    async handleTwitterInteractions() {
        elizaLogger.log("Checking Twitter interactions");

        const twitterUsername = this.client.profile.username;
        try {
            const tweetCandidates = (
                await this.client.fetchSearchTweets(
                    `@${twitterUsername}`,
                    20,
                    SearchMode.Latest
                )
            ).tweets;

            const uniqueTweetCandidates = [...new Set(tweetCandidates)]
                .sort((a, b) => a.id.localeCompare(b.id))
                .filter((tweet) => tweet.userId !== this.client.profile.id);

            for (const tweet of uniqueTweetCandidates) {
                if (
                    !this.client.lastCheckedTweetId ||
                    BigInt(tweet.id) > this.client.lastCheckedTweetId
                ) {
                    // Skip if we can't reply to this user or in this thread
                    if (!this.canReplyToUser(tweet.userId) || 
                        !this.canReplyInThread(tweet.conversationId)) {
                        elizaLogger.log("Skipping tweet due to rate limits", tweet.permanentUrl);
                        continue;
                    }

                    // Apply random probability check
                    if (Math.random() > this.limits.replyProbability) {
                        elizaLogger.log("Skipping tweet due to probability check", tweet.permanentUrl);
                        continue;
                    }

                    elizaLogger.log("New Tweet found", tweet.permanentUrl);

                    const roomId = stringToUuid(
                        tweet.conversationId + "-" + this.runtime.agentId
                    );

                    const userIdUUID =
                        tweet.userId === this.client.profile.id
                            ? this.runtime.agentId
                            : stringToUuid(tweet.userId!);

                    await this.runtime.ensureConnection(
                        userIdUUID,
                        roomId,
                        tweet.username,
                        tweet.name,
                        "twitter"
                    );

                    const thread = await buildConversationThread(
                        tweet,
                        this.client
                    );

                    const message = {
                        content: { text: tweet.text },
                        agentId: this.runtime.agentId,
                        userId: userIdUUID,
                        roomId,
                    };

                    const result = await this.handleTweet({
                        tweet,
                        message,
                        thread,
                    });

                    if (result?.action === "RESPOND") {
                        // Update tracking only if we actually replied
                        this.updateInteractionTracking(tweet.userId, tweet.conversationId);
                    }

                    this.client.lastCheckedTweetId = BigInt(tweet.id);
                }
            }

            await this.client.cacheLatestCheckedTweetId();
            elizaLogger.log("Finished checking Twitter interactions");

        } catch (error) {
            elizaLogger.error("Error handling Twitter interactions:", error);
        }
    }

    private async handleTweet({
        tweet,
        message,
        thread,
    }: {
        tweet: Tweet;
        message: Memory;
        thread: Tweet[];
    }) {
        if (tweet.userId === this.client.profile.id) {
            elizaLogger.log("Skipping tweet from bot itself", tweet.id);
            return { text: "", action: "IGNORE" };
        }

        if (!message.content.text) {
            elizaLogger.log("Skipping Tweet with no text", tweet.id);
            return { text: "", action: "IGNORE" };
        }

        elizaLogger.log("Processing Tweet: ", tweet.id);
        const formatTweet = (tweet: Tweet) => {
            return `  ID: ${tweet.id}
  From: ${tweet.name} (@${tweet.username})
  Text: ${tweet.text}`;
        };
        const currentPost = formatTweet(tweet);

        let homeTimeline: Tweet[] = [];
        const cachedTimeline = await this.client.getCachedTimeline();
        
        if (cachedTimeline) {
            homeTimeline = cachedTimeline;
        } else {
            homeTimeline = await this.client.fetchHomeTimeline(50);
            await this.client.cacheTimeline(homeTimeline);
        }

        elizaLogger.debug("Thread: ", thread);
        const formattedConversation = thread
            .map(
                (tweet) => `@${tweet.username} (${new Date(
                    tweet.timestamp * 1000
                ).toLocaleString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                    month: "short",
                    day: "numeric",
                })}):
        ${tweet.text}`
            )
            .join("\n\n");

        elizaLogger.debug("formattedConversation: ", formattedConversation);

        const formattedHomeTimeline =
            `# ${this.runtime.character.name}'s Home Timeline\n\n` +
            homeTimeline
                .map((tweet) => {
                    return `ID: ${tweet.id}\nFrom: ${tweet.name} (@${tweet.username})${tweet.inReplyToStatusId ? ` In reply to: ${tweet.inReplyToStatusId}` : ""}\nText: ${tweet.text}\n---\n`;
                })
                .join("\n");

        let state = await this.runtime.composeState(message, {
            twitterClient: this.client.twitterClient,
            twitterUserName: this.runtime.getSetting("TWITTER_USERNAME"),
            currentPost,
            formattedConversation,
            timeline: formattedHomeTimeline,
        });

        // Check if the tweet exists, save if it doesn't
        const tweetId = stringToUuid(tweet.id + "-" + this.runtime.agentId);
        const tweetExists = await this.runtime.messageManager.getMemoryById(tweetId);

        if (!tweetExists) {
            elizaLogger.log("tweet does not exist, saving");
            const userIdUUID = stringToUuid(tweet.userId as string);
            const roomId = stringToUuid(tweet.conversationId);

            const message = {
                id: tweetId,
                agentId: this.runtime.agentId,
                content: {
                    text: tweet.text,
                    url: tweet.permanentUrl,
                    inReplyTo: tweet.inReplyToStatusId
                        ? stringToUuid(
                              tweet.inReplyToStatusId +
                                  "-" +
                                  this.runtime.agentId
                          )
                        : undefined,
                },
                userId: userIdUUID,
                roomId,
                createdAt: tweet.timestamp * 1000,
            };
            this.client.saveRequestMessage(message, state);
        }

        const shouldRespondContext = composeContext({
            state,
            template:
                this.runtime.character.templates?.twitterShouldRespondTemplate ||
                this.runtime.character?.templates?.shouldRespondTemplate ||
                twitterShouldRespondTemplate,
        });

        const shouldRespond = await generateShouldRespond({
            runtime: this.runtime,
            context: shouldRespondContext,
            modelClass: ModelClass.MEDIUM,
        });

        if (shouldRespond !== "RESPOND") {
            elizaLogger.log("Not responding to message");
            return { text: "Response Decision:", action: shouldRespond };
        }

        const context = composeContext({
            state,
            template:
                this.runtime.character.templates?.twitterMessageHandlerTemplate ||
                this.runtime.character?.templates?.messageHandlerTemplate ||
                twitterMessageHandlerTemplate,
        });

        elizaLogger.debug("Interactions prompt:\n" + context);

        const response = await generateMessageResponse({
            runtime: this.runtime,
            context,
            modelClass: ModelClass.MEDIUM,
        });

        const stringId = stringToUuid(tweet.id + "-" + this.runtime.agentId);
        response.inReplyTo = stringId;
        response.text = response.text.replace(/^['"](.*)['"]$/, "$1");

        if (response.text) {
            try {
                const callback: HandlerCallback = async (response: Content) => {
                    const memories = await sendTweet(
                        this.client,
                        response,
                        message.roomId,
                        this.runtime.getSetting("TWITTER_USERNAME"),
                        tweet.id
                    );
                    return memories;
                };

                const responseMessages = await callback(response);

                state = await this.runtime.updateRecentMessageState(state);

                for (const responseMessage of responseMessages) {
                    if (responseMessage === responseMessages[responseMessages.length - 1]) {
                        responseMessage.content.action = response.action;
                    } else {
                        responseMessage.content.action = "CONTINUE";
                    }
                    await this.runtime.messageManager.createMemory(responseMessage);
                }

                state = await this.runtime.updateRecentMessageState(state);

                await this.runtime.evaluate(message, state);

                await this.runtime.processActions(
                    message,
                    responseMessages,
                    state
                );

                const responseInfo = `Context:\n\n${context}\n\nSelected Post: ${tweet.id} - ${tweet.username}: ${tweet.text}\nAgent's Output:\n${response.text}`;

                await this.runtime.cacheManager.set(
                    `twitter/tweet_generation_${tweet.id}.txt`,
                    responseInfo
                );
                
                await wait();

                return { text: response.text, action: "RESPOND" };
            } catch (error) {
                elizaLogger.error(`Error sending response tweet: ${error}`);
                return { text: "", action: "ERROR" };
            }
        }
        return { text: "", action: "IGNORE" };
    }

    async buildConversationThread(
        tweet: Tweet,
        maxReplies: number = this.limits.maxRepliesPerThread
    ): Promise<Tweet[]> {
        const thread: Tweet[] = [];
        const visited: Set<string> = new Set();

        async function processThread(currentTweet: Tweet, depth: number = 0) {
            elizaLogger.log("Processing tweet:", {
                id: currentTweet.id,
                inReplyToStatusId: currentTweet.inReplyToStatusId,
                depth: depth,
            });

            if (!currentTweet) {
                elizaLogger.log("No current tweet found for thread building");
                return;
            }

            if (depth >= maxReplies) {
                elizaLogger.log("Reached maximum reply depth", depth);
                return;
            }

            // Handle memory storage
            const memory = await this.runtime.messageManager.getMemoryById(
                stringToUuid(currentTweet.id + "-" + this.runtime.agentId)
            );
            if (!memory) {
                const roomId = stringToUuid(
                    currentTweet.conversationId + "-" + this.runtime.agentId
                );
                const userId = stringToUuid(currentTweet.userId);

                await this.runtime.ensureConnection(
                    userId,
                    roomId,
                    currentTweet.username,
                    currentTweet.name,
                    "twitter"
                );

                this.runtime.messageManager.createMemory({
                    id: stringToUuid(currentTweet.id + "-" + this.runtime.agentId),
                    agentId: this.runtime.agentId,
                    content: {
                        text: currentTweet.text,
                        source: "twitter",
                        url: currentTweet.permanentUrl,
                        inReplyTo: currentTweet.inReplyToStatusId
                            ? stringToUuid(currentTweet.inReplyToStatusId + "-" + this.runtime.agentId)
                            : undefined,
                    },
                    createdAt: currentTweet.timestamp * 1000,
                    roomId,
                    userId: currentTweet.userId === this.client.profile.id
                        ? this.runtime.agentId
                        : stringToUuid(currentTweet.userId),
                    embedding: embeddingZeroVector,
                });
            }

            if (visited.has(currentTweet.id)) {
                elizaLogger.log("Already visited tweet:", currentTweet.id);
                return;
            }

            visited.add(currentTweet.id);
            thread.unshift(currentTweet);

            elizaLogger.debug("Current thread state:", {
                length: thread.length,
                currentDepth: depth,
                tweetId: currentTweet.id,
            });

            if (currentTweet.inReplyToStatusId) {
                elizaLogger.log("Fetching parent tweet:", currentTweet.inReplyToStatusId);
                try {
                    const parentTweet = await this.client.twitterClient.getTweet(
                        currentTweet.inReplyToStatusId
                    );

                    if (parentTweet) {
                        elizaLogger.log("Found parent tweet:", {
                            id: parentTweet.id,
                            text: parentTweet.text?.slice(0, 50),
                        });
                        await processThread.bind(this)(parentTweet, depth + 1);
                    } else {
                        elizaLogger.log("No parent tweet found for:", currentTweet.inReplyToStatusId);
                    }
                } catch (error) {
                    elizaLogger.log("Error fetching parent tweet:", {
                        tweetId: currentTweet.inReplyToStatusId,
                        error,
                    });
                }
            } else {
                elizaLogger.log("Reached end of reply chain at:", currentTweet.id);
            }
        }

        // Need to bind this context for the inner function
        await processThread.bind(this)(tweet, 0);

        elizaLogger.debug("Final thread built:", {
            totalTweets: thread.length,
            tweetIds: thread.map((t) => ({
                id: t.id,
                text: t.text?.slice(0, 50),
            })),
        });

        return thread;
    }
}
