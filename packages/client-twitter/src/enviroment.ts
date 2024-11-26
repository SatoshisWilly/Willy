// environment.ts

import { z } from "zod";
import { IAgentRuntime, elizaLogger } from "@ai16z/eliza";

export interface SpecialInteraction {
    handle: string;
    topics: string[];
    templates: string[];
    probability: number;
}

const specialInteractionSchema = z.object({
    handle: z.string(),
    topics: z.array(z.string()),
    templates: z.array(z.string()),
    probability: z.number().min(0).max(1)
});

export const twitterEnvSchema = z.object({
    TWITTER_DRY_RUN: z
        .string()
        .transform((val) => val.toLowerCase() === "true"),
    TWITTER_USERNAME: z.string().min(1, "Twitter username is required"),
    TWITTER_PASSWORD: z.string().min(1, "Twitter password is required"),
    TWITTER_EMAIL: z.string().email("Valid Twitter email is required"),
    TWITTER_COOKIES: z.string().optional(),
    TWITTER_SPECIAL_INTERACTIONS: z.string()
        .optional()
        .transform((val) => {
            if (!val) return {};
            try {
                const parsed = JSON.parse(val);
                const validated: Record<string, SpecialInteraction> = {};
                
                Object.entries(parsed).forEach(([key, value]) => {
                    try {
                        validated[key] = specialInteractionSchema.parse(value);
                    } catch (e) {
                        elizaLogger.warn(`Invalid special interaction config for ${key}:`, e);
                    }
                });
                
                return validated;
            } catch (e) {
                elizaLogger.error("Error parsing TWITTER_SPECIAL_INTERACTIONS:", e);
                return {};
            }
        }),
    TWITTER_SPECIAL_INTERACTION_COOLDOWN: z
        .string()
        .optional()
        .transform((val) => parseInt(val || '86400000')), // 24 hours default
});

export type TwitterConfig = z.infer<typeof twitterEnvSchema>;

export async function validateTwitterConfig(
    runtime: IAgentRuntime
): Promise<TwitterConfig> {
    try {
        const config = {
            TWITTER_DRY_RUN:
                runtime.getSetting("TWITTER_DRY_RUN") ||
                process.env.TWITTER_DRY_RUN,
            TWITTER_USERNAME:
                runtime.getSetting("TWITTER_USERNAME") ||
                process.env.TWITTER_USERNAME,
            TWITTER_PASSWORD:
                runtime.getSetting("TWITTER_PASSWORD") ||
                process.env.TWITTER_PASSWORD,
            TWITTER_EMAIL:
                runtime.getSetting("TWITTER_EMAIL") ||
                process.env.TWITTER_EMAIL,
            TWITTER_COOKIES:
                runtime.getSetting("TWITTER_COOKIES") ||
                process.env.TWITTER_COOKIES,
            TWITTER_SPECIAL_INTERACTIONS:
                runtime.getSetting("TWITTER_SPECIAL_INTERACTIONS") ||
                process.env.TWITTER_SPECIAL_INTERACTIONS,
            TWITTER_SPECIAL_INTERACTION_COOLDOWN:
                runtime.getSetting("TWITTER_SPECIAL_INTERACTION_COOLDOWN") ||
                process.env.TWITTER_SPECIAL_INTERACTION_COOLDOWN,
        };

        return twitterEnvSchema.parse(config);
    } catch (error) {
        if (error instanceof z.ZodError) {
            const errorMessages = error.errors
                .map((err) => `${err.path.join(".")}: ${err.message}`)
                .join("\n");
            throw new Error(
                `Twitter configuration validation failed:\n${errorMessages}`
            );
        }
        throw error;
    }
}
