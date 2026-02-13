
import { NextRequest, NextResponse } from "next/server";
import axios from "axios";

// Environment variables (set these in .env.local)
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;

export const maxDuration = 60; // Allow time for large response payloads

export async function GET(req: NextRequest) {
    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
        return NextResponse.json(
            { error: "Server misconfigured: Missing RunPod credentials." },
            { status: 500 }
        );
    }

    const { searchParams } = new URL(req.url);
    const id = searchParams.get("id");

    if (!id) {
        return NextResponse.json(
            { error: "Missing 'id' query parameter." },
            { status: 400 }
        );
    }

    try {
        const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${id}`;

        const response = await axios.get(url, {
            headers: {
                Authorization: `Bearer ${RUNPOD_API_KEY}`,
            },
            timeout: 30000,
        });

        const data = response.data;

        // RunPod response structure:
        // { id, status: "IN_PROGRESS" | "COMPLETED" | "FAILED", output: { ... }, error: ... }

        if (data.status === "FAILED") {
            return NextResponse.json({ status: "FAILED", error: data.error || "Unknown RunPod error" });
        }

        if (data.status === "COMPLETED") {
            return NextResponse.json({ status: "COMPLETED", output: data.output });
        }

        // Return IN_PROGRESS or QUEUED
        return NextResponse.json({ status: data.status });

    } catch (error: any) {
        console.error("Status Check Error:", error.response?.data || error.message);
        return NextResponse.json(
            { error: "Failed to check status", details: error.response?.data || error.message },
            { status: 500 }
        );
    }
}
