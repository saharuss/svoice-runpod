import { NextResponse } from "next/server";

const VM_METRICS_URL = "http://34.55.45.65:8080/api/metrics";

export async function GET() {
    try {
        const res = await fetch(VM_METRICS_URL, {
            cache: "no-store",
            signal: AbortSignal.timeout(8000),
        });

        if (!res.ok) {
            return NextResponse.json(
                { error: "VM metrics server returned " + res.status },
                { status: 502 }
            );
        }

        const data = await res.json();
        return NextResponse.json(data, {
            headers: { "Cache-Control": "no-cache, no-store" },
        });
    } catch (e: unknown) {
        const message = e instanceof Error ? e.message : "Unknown error";
        return NextResponse.json(
            { error: "Cannot reach VM metrics server", detail: message },
            { status: 502 }
        );
    }
}
