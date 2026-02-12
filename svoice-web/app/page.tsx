
"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import {
  UploadCloud,
  Music,
  Loader2,
  AlertCircle,
  Download,
  ChevronDown,
  ChevronRight,
  Mic,
  AudioLines,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Track {
  speaker: number;
  audio_base64: string;
  confidence: number;
  is_main: boolean;
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  let color = "bg-red-500/20 text-red-300 border-red-500/30";
  if (pct >= 60) color = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
  else if (pct >= 30) color = "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";

  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${color}`}>
      {pct}%
    </span>
  );
}

function TrackCard({ track }: { track: Track }) {
  return (
    <div className="glass p-4 rounded-xl flex items-center justify-between gap-4 group hover:bg-white/10 transition-colors">
      <div className="flex items-center gap-4">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${track.is_main
              ? "bg-purple-500/30 text-purple-300"
              : "bg-white/5 text-white/30"
            }`}
        >
          {track.speaker}
        </div>
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <p className="font-medium text-white/90">Speaker {track.speaker}</p>
            <ConfidenceBadge confidence={track.confidence} />
          </div>
          <p className="text-xs text-white/40">
            {track.is_main ? "Main Voice" : "Background / Noise"}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <audio
          controls
          src={`data:audio/wav;base64,${track.audio_base64}`}
          className="w-full max-w-[200px] h-10 opacity-80 hover:opacity-100 transition-opacity"
        />
        <button
          onClick={() => {
            const link = document.createElement("a");
            link.href = `data:audio/wav;base64,${track.audio_base64}`;
            link.download = `speaker_${track.speaker}.wav`;
            link.click();
          }}
          className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white/80 hover:text-white"
          title="Download Track"
        >
          <Download className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showOtherTracks, setShowOtherTracks] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      setFile(acceptedFiles[0]);
      setTracks(null);
      setError(null);
      setShowOtherTracks(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".wav", ".mp3", ".m4a", ".flac"],
    },
    maxFiles: 1,
  });

  const processAudio = async () => {
    if (!file) return;

    if (file.size > 20 * 1024 * 1024) {
      setError("File too large. Please use a file smaller than 20MB for the demo.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setShowOtherTracks(false);

    try {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = async () => {
        const base64String = (reader.result as string).split(",")[1];

        try {
          const response = await axios.post("/api/separate", {
            audio_base64: base64String,
            sample_rate: 16000,
          });

          if (response.data.separated_tracks) {
            setTracks(response.data.separated_tracks);
          } else {
            setError("No tracks returned from server.");
          }
        } catch (err: any) {
          console.error("Upload error:", err);
          if (err.response?.status === 413) {
            setError("File is too large for the server to process.");
          } else {
            setError(
              err.response?.data?.error ||
              "Failed to process audio. Check console for details."
            );
          }
        } finally {
          setIsProcessing(false);
        }
      };

      reader.onerror = () => {
        console.error("FileReader Error:", reader.error);
        setError(
          `Failed to read file: ${reader.error?.message || "Unknown error"}`
        );
        setIsProcessing(false);
      };
    } catch (err) {
      console.error("Unexpected error:", err);
      setError("An unexpected error occurred.");
      setIsProcessing(false);
    }
  };

  const mainTracks = tracks?.filter((t) => t.is_main) ?? [];
  const otherTracks = tracks?.filter((t) => !t.is_main) ?? [];

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 sm:p-24 relative overflow-hidden">
      {/* Background Orbs */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl translate-x-1/2 translate-y-1/2" />

      <div className="z-10 w-full max-w-3xl space-y-12">
        <header className="text-center space-y-4">
          <h1 className="text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            svoice
          </h1>
          <p className="text-lg text-white/60">
            State-of-the-art Speaker Voice Separation
          </p>
        </header>

        {/* Upload Zone */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div
            {...getRootProps()}
            className={`
              glass rounded-3xl p-12 text-center cursor-pointer transition-all duration-300 border-2
              ${isDragActive ? "border-purple-400 bg-purple-400/10 scale-[1.02]" : "border-white/5 hover:border-white/20"}
              ${file ? "ring-2 ring-purple-500/50" : ""}
            `}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-white/5 rounded-full">
                {file ? (
                  <Music className="w-10 h-10 text-purple-400" />
                ) : (
                  <UploadCloud className="w-10 h-10 text-white/40" />
                )}
              </div>
              <div className="space-y-2">
                <p className="text-xl font-medium text-white/90">
                  {file ? file.name : "Drop your audio file here"}
                </p>
                <p className="text-sm text-white/40">
                  {file
                    ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                    : "WAV, MP3, FLAC • Max 20MB"}
                </p>
              </div>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={processAudio}
              disabled={!file || isProcessing}
              className={`
                px-8 py-3 rounded-full font-medium text-lg transition-all duration-300 flex items-center gap-2
                ${!file || isProcessing
                  ? "bg-white/5 text-white/20 cursor-not-allowed"
                  : "bg-white text-black hover:scale-105 hover:shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                }
              `}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Separating Voices...
                </>
              ) : (
                "Start Separation"
              )}
            </button>
          </div>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-red-500/10 border border-red-500/20 text-red-200 p-4 rounded-xl flex items-center gap-3"
            >
              <AlertCircle className="w-5 h-5" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {tracks && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              {/* ── Main Voices Section ── */}
              {mainTracks.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 pl-2 border-l-4 border-emerald-500">
                    <Mic className="w-5 h-5 text-emerald-400" />
                    <h2 className="text-2xl font-semibold text-white/90">
                      Main Voices
                    </h2>
                    <span className="text-sm text-white/40">
                      ({mainTracks.length} detected)
                    </span>
                  </div>
                  <div className="grid gap-3">
                    {mainTracks.map((track) => (
                      <TrackCard key={track.speaker} track={track} />
                    ))}
                  </div>
                </div>
              )}

              {/* ── Other Tracks Section ── */}
              {otherTracks.length > 0 && (
                <div className="space-y-3">
                  <button
                    onClick={() => setShowOtherTracks(!showOtherTracks)}
                    className="flex items-center gap-3 pl-2 border-l-4 border-white/20 hover:border-white/40 transition-colors w-full text-left group"
                  >
                    {showOtherTracks ? (
                      <ChevronDown className="w-4 h-4 text-white/40 group-hover:text-white/60" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-white/60" />
                    )}
                    <AudioLines className="w-4 h-4 text-white/30" />
                    <h3 className="text-lg font-medium text-white/50 group-hover:text-white/70 transition-colors">
                      Other Tracks
                    </h3>
                    <span className="text-sm text-white/30">
                      ({otherTracks.length})
                    </span>
                  </button>
                  <AnimatePresence>
                    {showOtherTracks && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="grid gap-3 overflow-hidden"
                      >
                        {otherTracks.map((track) => (
                          <TrackCard key={track.speaker} track={track} />
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {/* Edge case: no main voices detected */}
              {mainTracks.length === 0 && (
                <div className="text-center text-white/40 py-4">
                  <p>No clear main voices detected. All tracks are shown below.</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}

