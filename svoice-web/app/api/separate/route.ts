
import { NextRequest, NextResponse } from "next/server";
import axios from "axios";

// Environment variables (set these in .env.local)
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;

export const maxDuration = 60; // Allow 60s for processing (Vercel/Next.js limit)

export async function POST(req: NextRequest) {
    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
        return NextResponse.json(
            { error: "Server misconfigured: Missing RunPod credentials." },
            { status: 500 }
        );
    }

    try {
        const body = await req.json();
        const { audio_base64, sample_rate, target_audio_base64 } = body;

        if (!audio_base64) {
            return NextResponse.json(
                { error: "Missing audio_base64 in request." },
                { status: 400 }
            );
        }

        // Call RunPod Async Endpoint (run) instead of Sync (runsync)
        const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`;

        console.log(`Submitting job to RunPod: ${url}`);

        const input: any = {
            audio_base64,
            sample_rate: sample_rate || 16000,
        };
        if (target_audio_base64) {
            input.target_audio_base64 = target_audio_base64;
        }

        const response = await axios.post(
            url,
            { input },
            {
                headers: {
                    Authorization: `Bearer ${RUNPOD_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 10000, // Short timeout for job submission
            }
        );

        const data = response.data;

        if (!data.id) {
            console.error("RunPod submission failed:", data);
            return NextResponse.json({ error: "RunPod submission failed", details: data }, { status: 502 });
        }

        // Return the Job ID for polling
        return NextResponse.json({ id: data.id, status: "IN_PROGRESS" });

    } catch (error: any) {
        console.error("API Error:", error.response?.data || error.message);
        return NextResponse.json(
            { error: "Failed to process audio", details: error.response?.data || error.message },
            { status: 500 }
        );
    }
}
