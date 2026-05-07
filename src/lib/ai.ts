import "server-only";

import { AzureOpenAI } from "openai";
import createClient from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import { getRequiredEnv } from "@/lib/env";

export type AIProviderId =
  | "openai"
  | "anthropic"
  | "google"
  | "azure-openai"
  | "azure-ai"
  | "openrouter"
  | "ollama";

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface GenerateChatCompletionInput {
  provider?: AIProviderId | string;
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  apiKey?: string;
  baseUrl?: string;
}

interface ChatProvider {
  id: AIProviderId;
  defaultModel: string;
  generate(input: RequiredModelInput): Promise<string>;
}

type RequiredModelInput = GenerateChatCompletionInput & {
  provider: AIProviderId;
  model: string;
  temperature: number;
  maxTokens: number;
};

interface OpenAICompatibleResponse {
  choices?: {
    message?: {
      content?: string | null;
    };
    text?: string;
  }[];
  error?: {
    message?: string;
  };
}

interface AnthropicResponse {
  content?: {
    type: string;
    text?: string;
  }[];
  error?: {
    message?: string;
  };
}

interface GoogleResponse {
  candidates?: {
    content?: {
      parts?: {
        text?: string;
      }[];
    };
  }[];
  error?: {
    message?: string;
  };
}

const PROVIDER_IDS: AIProviderId[] = [
  "openai",
  "anthropic",
  "google",
  "azure-openai",
  "azure-ai",
  "openrouter",
  "ollama",
];

const AZURE_AI_ENDPOINT_ENV_BY_MODEL: Record<string, string> = {
  "grok-3-mini": "GROK_3_MINI_ENDPOINT",
  "deepseek-v3": "DEEPSEEK_V3_ENDPOINT",
  "grok-3": "GROK_3_ENDPOINT",
};

function getEnv(...names: string[]) {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  return "";
}

function normalizeProvider(provider?: string | null): AIProviderId | null {
  if (!provider) return null;
  const normalized = provider.trim().toLowerCase();
  return PROVIDER_IDS.includes(normalized as AIProviderId)
    ? (normalized as AIProviderId)
    : null;
}

function configuredProvider() {
  const value = process.env.AI_PROVIDER?.trim();
  if (!value) return null;

  const provider = normalizeProvider(value);
  if (!provider) {
    throw new Error(
      `Unsupported AI_PROVIDER "${value}". Valid providers: ${PROVIDER_IDS.join(", ")}.`
    );
  }

  return provider;
}

function isAzureAIModel(model: string) {
  return model in AZURE_AI_ENDPOINT_ENV_BY_MODEL;
}

function inferProvider(model: string): AIProviderId {
  const provider = configuredProvider();
  if (provider) return provider;

  if (isAzureAIModel(model) || getEnv("AZURE_AI_API_KEY")) return "azure-ai";
  if (getEnv("AZURE_OPENAI_KEY") && getEnv("AZURE_OPENAI_ENDPOINT")) {
    return "azure-openai";
  }
  if (getEnv("OPENAI_API_KEY")) return "openai";
  if (getEnv("ANTHROPIC_API_KEY", "AZURE_ANTHROPIC_API_KEY")) return "anthropic";
  if (getEnv("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY")) return "google";
  if (getEnv("OPENROUTER_API_KEY")) return "openrouter";
  if (getEnv("OLLAMA_BASE_URL")) return "ollama";

  return "openai";
}

function requireValue(value: string, message: string) {
  if (!value) throw new Error(message);
  return value;
}

function getAzureAIEndpoint(model: string) {
  const endpointEnvName = AZURE_AI_ENDPOINT_ENV_BY_MODEL[model];
  return getEnv(endpointEnvName || "", "AZURE_AI_ENDPOINT");
}

function shouldOmitTemperature(model: string) {
  const normalized = model.toLowerCase();
  return (
    normalized.startsWith("gpt-5") ||
    normalized.startsWith("gpt-4.1") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("o4") ||
    normalized.includes("/gpt-5") ||
    normalized.includes("/gpt-4.1")
  );
}

function shouldUseCompletionTokenLimit(provider: AIProviderId, model: string) {
  if (provider !== "openai") return false;

  const normalized = model.toLowerCase();
  return (
    normalized.startsWith("gpt-5") ||
    normalized.startsWith("gpt-4.1") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("o4")
  );
}

async function readError(response: Response) {
  const text = await response.text().catch(() => "");
  return text || `${response.status} ${response.statusText}`;
}

function extractOpenAICompatibleText(data: OpenAICompatibleResponse) {
  const choice = data.choices?.[0];
  const text = choice?.message?.content ?? choice?.text;
  if (text) return text;
  throw new Error(`Unexpected AI response: ${JSON.stringify(data).slice(0, 500)}`);
}

async function generateWithOpenAICompatible({
  input,
  apiKey,
  baseUrl,
  headers,
  requireApiKey = true,
}: {
  input: RequiredModelInput;
  apiKey: string;
  baseUrl: string;
  headers?: Record<string, string>;
  requireApiKey?: boolean;
}) {
  if (requireApiKey) {
    requireValue(apiKey, `${input.provider} API key is not configured.`);
  }

  const normalizedBaseUrl = baseUrl.replace(/\/$/, "");
  const requestHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...headers,
  };

  if (apiKey) {
    requestHeaders.Authorization = `Bearer ${apiKey}`;
  }

  const body: Record<string, unknown> = {
    model: input.model,
    messages: input.messages,
  };
  body[
    shouldUseCompletionTokenLimit(input.provider, input.model)
      ? "max_completion_tokens"
      : "max_tokens"
  ] = input.maxTokens;

  if (!shouldOmitTemperature(input.model)) {
    body.temperature = input.temperature;
  }

  const response = await fetch(`${normalizedBaseUrl}/chat/completions`, {
    method: "POST",
    headers: requestHeaders,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`${input.provider} error (${response.status}): ${await readError(response)}`);
  }

  return extractOpenAICompatibleText(
    (await response.json()) as OpenAICompatibleResponse
  );
}

async function generateWithAnthropic(input: RequiredModelInput) {
  const apiKey =
    input.apiKey || getEnv("ANTHROPIC_API_KEY", "AZURE_ANTHROPIC_API_KEY");
  const baseUrl = (
    input.baseUrl ||
    getEnv("ANTHROPIC_BASE_URL", "AZURE_ANTHROPIC_ENDPOINT") ||
    "https://api.anthropic.com"
  ).replace(/\/$/, "");

  requireValue(apiKey, "Anthropic API key is not configured.");

  const system = input.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");
  const messages = input.messages
    .filter((message) => message.role !== "system")
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.content,
    }));

  const response = await fetch(`${baseUrl}/v1/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: input.model,
      max_tokens: input.maxTokens,
      temperature: input.temperature,
      ...(system ? { system } : {}),
      messages,
    }),
  });

  if (!response.ok) {
    throw new Error(`Anthropic error (${response.status}): ${await readError(response)}`);
  }

  const data = (await response.json()) as AnthropicResponse;
  const text = data.content
    ?.filter((block) => block.type === "text")
    .map((block) => block.text || "")
    .join("");

  if (text) return text;
  throw new Error(`Unexpected Anthropic response: ${JSON.stringify(data).slice(0, 500)}`);
}

async function generateWithGoogle(input: RequiredModelInput) {
  const apiKey =
    input.apiKey || getEnv("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY");
  const baseUrl = (
    input.baseUrl || "https://generativelanguage.googleapis.com/v1beta"
  ).replace(/\/$/, "");

  requireValue(apiKey, "Google Generative AI API key is not configured.");

  const systemText = input.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");
  const contents = input.messages
    .filter((message) => message.role !== "system")
    .map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    }));

  const response = await fetch(
    `${baseUrl}/models/${encodeURIComponent(input.model)}:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...(systemText ? { systemInstruction: { parts: [{ text: systemText }] } } : {}),
        contents,
        generationConfig: {
          temperature: input.temperature,
          maxOutputTokens: input.maxTokens,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Google AI error (${response.status}): ${await readError(response)}`);
  }

  const data = (await response.json()) as GoogleResponse;
  const text = data.candidates?.[0]?.content?.parts
    ?.map((part) => part.text || "")
    .join("");

  if (text) return text;
  throw new Error(`Unexpected Google AI response: ${JSON.stringify(data).slice(0, 500)}`);
}

async function generateWithAzureOpenAI(input: RequiredModelInput) {
  const client = new AzureOpenAI({
    endpoint: input.baseUrl || getRequiredEnv("AZURE_OPENAI_ENDPOINT"),
    apiKey: input.apiKey || getRequiredEnv("AZURE_OPENAI_KEY"),
    apiVersion: getRequiredEnv("AZURE_OPENAI_API_VERSION"),
    deployment: getRequiredEnv("AZURE_OPENAI_DEPLOYMENT"),
  });

  const response = await client.chat.completions.create({
    messages: input.messages,
    max_completion_tokens: input.maxTokens,
    model: input.model,
  });

  const text = response?.choices[0]?.message?.content;
  if (text) return text;
  throw new Error("Unexpected Azure OpenAI response.");
}

async function generateWithAzureAI(input: RequiredModelInput) {
  const endpoint = requireValue(
    input.baseUrl || getAzureAIEndpoint(input.model),
    `No Azure AI endpoint configured for model "${input.model}".`
  );
  const client = createClient(
    endpoint,
    new AzureKeyCredential(input.apiKey || getRequiredEnv("AZURE_AI_API_KEY"))
  );

  const response = await client.path("/chat/completions").post({
    body: {
      messages: input.messages,
      max_tokens: input.maxTokens,
      temperature: input.temperature,
      top_p: 1,
      model: input.model,
    },
  });

  if (response.status !== "200") throw response.body;
  const body = response.body as {
    choices?: { message?: { content?: string } }[];
  };
  const text = body.choices?.[0]?.message?.content;
  if (text) return text;
  throw new Error("Unexpected Azure AI response.");
}

const providers: Record<AIProviderId, ChatProvider> = {
  openai: {
    id: "openai",
    defaultModel: "gpt-4.1",
    generate: (input) =>
      generateWithOpenAICompatible({
        input,
        apiKey: input.apiKey || getEnv("OPENAI_API_KEY"),
        baseUrl: input.baseUrl || getEnv("OPENAI_BASE_URL") || "https://api.openai.com/v1",
      }),
  },
  anthropic: {
    id: "anthropic",
    defaultModel: "claude-3-5-sonnet-latest",
    generate: generateWithAnthropic,
  },
  google: {
    id: "google",
    defaultModel: "gemini-1.5-pro",
    generate: generateWithGoogle,
  },
  "azure-openai": {
    id: "azure-openai",
    defaultModel: "gpt-4.1",
    generate: generateWithAzureOpenAI,
  },
  "azure-ai": {
    id: "azure-ai",
    defaultModel: "grok-3-mini",
    generate: generateWithAzureAI,
  },
  openrouter: {
    id: "openrouter",
    defaultModel: "openai/gpt-4.1",
    generate: (input) =>
      generateWithOpenAICompatible({
        input,
        apiKey: input.apiKey || getEnv("OPENROUTER_API_KEY"),
        baseUrl: input.baseUrl || "https://openrouter.ai/api/v1",
        headers: {
          "HTTP-Referer": getEnv("OPENROUTER_SITE_URL", "NEXT_PUBLIC_APP_URL"),
          "X-Title": getEnv("OPENROUTER_APP_NAME") || "FlickBuddy",
        },
      }),
  },
  ollama: {
    id: "ollama",
    defaultModel: "llama3.1",
    generate: (input) =>
      generateWithOpenAICompatible({
        input,
        apiKey: input.apiKey || "",
        baseUrl: input.baseUrl || getEnv("OLLAMA_BASE_URL") || "http://localhost:11434/v1",
        requireApiKey: false,
      }),
  },
};

export function getDefaultAIProvider(model?: string) {
  return inferProvider(model || process.env.AI_MODEL || "");
}

export function getDefaultAIModel(fallback = "gpt-4.1") {
  const configuredModel = process.env.AI_MODEL?.trim();
  if (configuredModel) return configuredModel;

  const provider = getDefaultAIProvider(fallback);
  return providers[provider]?.defaultModel || fallback;
}

export async function generateChatCompletion(
  input: GenerateChatCompletionInput
): Promise<string> {
  const model = input.model || getDefaultAIModel();
  const provider =
    normalizeProvider(input.provider || null) || getDefaultAIProvider(model);
  const chatProvider = providers[provider];

  return chatProvider.generate({
    ...input,
    provider,
    model,
    temperature: input.temperature ?? 0.3,
    maxTokens: input.maxTokens ?? 16000,
  });
}

export async function generateChatCompletionWithModel(
  prompt: string,
  model: string = getDefaultAIModel(),
  temperature: number = 0.3,
  max_tokens: number = 16000
): Promise<string> {
  return generateChatCompletion({
    model,
    temperature,
    maxTokens: max_tokens,
    messages: [{ role: "user", content: prompt }],
  });
}
