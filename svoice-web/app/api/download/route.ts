import { NextRequest, NextResponse } from "next/server";

const VM_IP = "34.46.141.38";
const VM_PORT = 8080;

// Allowed files to download from the VM
const ALLOWED_FILES = ["best.th", "checkpoint.th", "history.json"];

export async function GET(req: NextRequest) {
    const file = req.nextUrl.searchParams.get("file");

    if (!file || !ALLOWED_FILES.includes(file)) {
        return NextResponse.json(
            { error: "Invalid file. Allowed: " + ALLOWED_FILES.join(", ") },
            { status: 400 }
        );
    }

    try {
        const res = await fetch(`http://${VM_IP}:${VM_PORT}/download/${file}`, {
            cache: "no-store",
            signal: AbortSignal.timeout(120_000), // 2 min timeout for large files
        });

        if (!res.ok) {
            return NextResponse.json(
                { error: `File not available (${res.status})` },
                { status: res.status === 404 ? 404 : 502 }
            );
        }

        const blob = await res.blob();
        return new NextResponse(blob, {
            headers: {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": `attachment; filename="${file}"`,
                "Content-Length": blob.size.toString(),
            },
        });
    } catch (e: unknown) {
        const message = e instanceof Error ? e.message : "Unknown error";
        return NextResponse.json(
            { error: "Cannot reach VM", detail: message },
            { status: 502 }
        );
    }
}
