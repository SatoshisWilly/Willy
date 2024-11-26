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
import { buildConversationThread, sendTweet } from "./utils.ts";
import { embeddingZeroVector } from "@ai16z/eliza";

export const twitterMessageHandlerTemplate = `
{{timeline}}

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

export const twitterShouldRespondTemplate = `
# INSTRUCTIONS: Determine if {{agentName}} (@{{twitterUserName}}) should respond to the message and participate in the conversation. Do not comment. Just respond with "true" or "false".

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

export class TwitterInteractionClient {
    client: ClientBase;
    runtime: IAgentRuntime;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    async start() {
        const handleTwitterInteractionsLoop = () => {
            this.handleTwitterInteractions();
            setTimeout(
                handleTwitterInteractionsLoop,
                (Math.floor(Math.random() * (5 - 2 + 1)) + 2) * 60 * 1000
            ); // Random interval between 2-5 minutes
        };
        handleTwitterInteractionsLoop();
    }

    async handleTwitterInteractions() {
        elizaLogger.log("Checking Twitter interactions");

        const twitterUsername = this.client.profile.username;

        try {
            // Fetch recent mentions
            const tweetCandidates = (
                await this.client.fetchSearchTweets(
                    `@${twitterUsername}`,
                    20,
                    SearchMode.Latest
                )
            ).tweets;

            const uniqueTweetCandidates = [...new Set(tweetCandidates)]
                .sort((a, b) => a.id.localeCompare(b.id))
                .filter(
                    (tweet) =>
                        tweet.userId !== this.client.profile.id && // Exclude self-tweets
                        !this.isSpam(tweet) &&
                        !(await this.hasRepliedToTweet(tweet.id)) // Exclude already replied tweets
                );

            for (const tweet of uniqueTweetCandidates) {
                if (
                    !this.client.lastCheckedTweetId ||
                    parseInt(tweet.id) > this.client.lastCheckedTweetId
                ) {
                    elizaLogger.log("New Tweet found", tweet.permanentUrl);

                    const thread = await this.buildConversationThread(tweet);

                    await this.handleTweet({
                        tweet,
                        message: {
                            content: { text: tweet.text },
                            agentId: this.runtime.agentId,
                            userId: stringToUuid(tweet.userId!),
                            roomId: stringToUuid(
                                tweet.conversationId + "-" + this.runtime.agentId
                            ),
                        },
                        thread,
                    });

                    // Mark tweet as replied after processing
                    await this.markTweetAsReplied(tweet.id);
                }
            }

            // Save the latest checked tweet ID
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
        if (tweet.userId === this.client.profile.id) return;

        elizaLogger.log("Processing Tweet: ", tweet.id);

        const shouldRespond = await this.shouldRespondToTweet(tweet, thread);

        if (!shouldRespond) {
            elizaLogger.log(`Not responding to tweet: ${tweet.id}`);
            return;
        }

        const response = await this.generateTweetResponse(tweet, thread);

        if (response.text) {
            try {
                await this.sendResponse(tweet, response);
                elizaLogger.log(`Responded to tweet: ${tweet.permanentUrl}`);
            } catch (error) {
                elizaLogger.error(`Error sending response tweet: ${error}`);
            }
        }
    }

    private async hasRepliedToTweet(tweetId: string): Promise<boolean> {
        const repliedKey = `twitter/replied/${tweetId}`;
        return !!(await this.runtime.cacheManager.get<boolean>(repliedKey));
    }

    private async markTweetAsReplied(tweetId: string): Promise<void> {
        const repliedKey = `twitter/replied/${tweetId}`;
        await this.runtime.cacheManager.set(repliedKey, true, { ttl: 30 * 24 * 60 * 60 }); // Cache for 30 days
    }

    private async shouldRespondToTweet(
        tweet: Tweet,
        thread: Tweet[]
    ): Promise<boolean> {
        const context = composeContext({
            state: await this.runtime.composeState(
                {
                    content: { text: tweet.text },
                    agentId: this.runtime.agentId,
                    userId: stringToUuid(tweet.userId!),
                    roomId: stringToUuid(
                        tweet.conversationId + "-" + this.runtime.agentId
                    ),
                },
                {
                    twitterUserName: this.client.profile.username,
                    formattedConversation: thread
                        .map((t) => `@${t.username}: ${t.text}`)
                        .join("\n"),
                }
            ),
            template: twitterShouldRespondTemplate,
        });

        const shouldRespond = await generateShouldRespond({
            runtime: this.runtime,
            context,
            modelClass: ModelClass.MEDIUM,
        });

        return shouldRespond === "RESPOND";
    }

    private async generateTweetResponse(tweet: Tweet, thread: Tweet[]) {
        const context = composeContext({
            state: await this.runtime.composeState(
                {
                    content: { text: tweet.text },
                    agentId: this.runtime.agentId,
                    userId: stringToUuid(tweet.userId!),
                    roomId: stringToUuid(
                        tweet.conversationId + "-" + this.runtime.agentId
                    ),
                },
                {
                    twitterUserName: this.client.profile.username,
                    formattedConversation: thread
                        .map((t) => `@${t.username}: ${t.text}`)
                        .join("\n"),
                }
            ),
            template: twitterMessageHandlerTemplate,
        });

        return await generateMessageResponse({
            runtime: this.runtime,
            context,
            modelClass: ModelClass.MEDIUM,
        });
    }

    private async sendResponse(tweet: Tweet, response: Content) {
        const memories = await sendTweet(
            this.client,
            response,
            stringToUuid(
                tweet.conversationId + "-" + this.runtime.agentId
            ),
            this.runtime.getSetting("TWITTER_USERNAME"),
            tweet.id
        );

        for (const memory of memories) {
            await this.runtime.messageManager.createMemory(memory);
        }
    }

    private isSpam(tweet: Tweet): boolean {
        const spamIndicators = [
            tweet.text.includes("promotion"),
            tweet.text.includes("DM for collab"),
            tweet.text.includes("buy followers"),
            tweet.username.includes("bot"),
            tweet.text.match(/http[s]?:\/\/[^\s]+/g)?.length > 2,
        ];

        const isNewAccount =
            new Date().getTime() - new Date(tweet.timestamp).getTime() <
            7 * 24 * 60 * 60 * 1000;

        return spamIndicators.some((indicator) => indicator) || isNewAccount;
    }
}
