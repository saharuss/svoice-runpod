
"use client";

import { useState, useCallback, useRef, useEffect } from "react";
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
  FileText,
  Mail,
  Play,
  Pause,
  Volume2,
  Users,
  Headphones,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface TimedWord {
  text: string;
  start: number;
  end: number;
}

interface Transcription {
  text: string;
  words: TimedWord[];
}

interface Track {
  speaker: number;
  audio_base64: string;
  confidence: number;
  is_main: boolean;
  transcription?: Transcription;
}

// ── Sample data ──────────────────────────────────────────────
const SAMPLES = [
  { file: "/samples/4people.wav", label: "4 Speakers", speakers: 4, duration: "7s" },
  { file: "/samples/5people.wav", label: "5 Speakers", speakers: 5, duration: "7s" },
  { file: "/samples/7people.wav", label: "7 Speakers", speakers: 7, duration: "10s" },
];

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

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ── Sample Card ──────────────────────────────────────────────
function SampleCard({
  sample,
  onTry,
}: {
  sample: (typeof SAMPLES)[number];
  onTry: (file: string) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onLoadedMetadata = () => setDuration(audio.duration);
    const onPlay = () => {
      setIsPlaying(true);
      const tick = () => {
        setCurrentTime(audio.currentTime);
        animRef.current = requestAnimationFrame(tick);
      };
      animRef.current = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setIsPlaying(false);
      cancelAnimationFrame(animRef.current);
    };
    const onEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      cancelAnimationFrame(animRef.current);
    };
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    if (audio.duration) setDuration(audio.duration);
    return () => {
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    isPlaying ? audio.pause() : audio.play();
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    const bar = progressRef.current;
    if (!audio || !bar || !duration) return;
    const rect = bar.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    audio.currentTime = (x / rect.width) * duration;
    setCurrentTime(audio.currentTime);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-5 space-y-4 hover:bg-white/10 transition-all duration-300 group"
    >
      <audio ref={audioRef} src={sample.file} preload="metadata" />

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/30 to-blue-500/30 flex items-center justify-center border border-purple-500/20">
          <Users className="w-5 h-5 text-purple-300" />
        </div>
        <div>
          <p className="font-semibold text-white/90">{sample.label}</p>
          <p className="text-xs text-white/40">
            {sample.duration} • Mixed audio
          </p>
        </div>
      </div>

      {/* Player */}
      <div
        className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300 ${isPlaying
          ? "bg-purple-500/10 border border-purple-500/20"
          : "bg-white/5 border border-white/5"
          }`}
      >
        <button
          onClick={togglePlay}
          className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-all duration-200 ${isPlaying
            ? "bg-purple-500 text-white shadow-[0_0_10px_rgba(168,85,247,0.4)]"
            : "bg-white/10 text-white/60 hover:bg-white/20 hover:text-white"
            }`}
        >
          {isPlaying ? (
            <Pause className="w-3.5 h-3.5" />
          ) : (
            <Play className="w-3.5 h-3.5 ml-0.5" />
          )}
        </button>
        <span className="text-xs font-mono text-white/40 w-8 shrink-0 text-right">
          {formatTime(currentTime)}
        </span>
        <div
          ref={progressRef}
          onClick={handleSeek}
          className="flex-1 h-1 bg-white/10 rounded-full cursor-pointer relative overflow-hidden"
        >
          <div
            className={`h-full rounded-full ${isPlaying
              ? "bg-gradient-to-r from-purple-500 to-blue-400"
              : "bg-white/25"
              }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-xs font-mono text-white/25 w-8 shrink-0">
          {formatTime(duration)}
        </span>
      </div>

      {/* Try It button */}
      <button
        onClick={() => onTry(sample.file)}
        className="w-full px-4 py-2.5 rounded-xl font-medium text-sm bg-gradient-to-r from-purple-600/80 to-blue-600/80 text-white hover:from-purple-500 hover:to-blue-500 transition-all duration-300 hover:shadow-[0_0_20px_rgba(168,85,247,0.3)] flex items-center justify-center gap-2"
      >
        <Headphones className="w-4 h-4" />
        Try Separation
      </button>
    </motion.div>
  );
}

// ── Track Card ───────────────────────────────────────────────
function TrackCard({ track }: { track: Track }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const hasWords =
    track.transcription?.words && track.transcription.words.length > 0;
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onLoadedMetadata = () => setDuration(audio.duration);
    const onPlay = () => {
      setIsPlaying(true);
      const tick = () => {
        setCurrentTime(audio.currentTime);
        animRef.current = requestAnimationFrame(tick);
      };
      animRef.current = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setIsPlaying(false);
      cancelAnimationFrame(animRef.current);
    };
    const onEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      cancelAnimationFrame(animRef.current);
    };

    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);

    if (audio.duration) setDuration(audio.duration);

    return () => {
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    const bar = progressRef.current;
    if (!audio || !bar || !duration) return;
    const rect = bar.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    audio.currentTime = (x / rect.width) * duration;
    setCurrentTime(audio.currentTime);
  };

  return (
    <div className="glass p-4 rounded-xl space-y-3 group hover:bg-white/10 transition-colors">
      <audio
        ref={audioRef}
        src={`data:audio/wav;base64,${track.audio_base64}`}
        preload="metadata"
      />

      {/* Header: speaker info + download */}
      <div className="flex items-center justify-between gap-4">
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
              <p className="font-medium text-white/90">
                Speaker {track.speaker}
              </p>
              <ConfidenceBadge confidence={track.confidence} />
            </div>
            <p className="text-xs text-white/40">
              {track.is_main ? "Main Voice • Enhanced" : "Background / Noise"}
            </p>
          </div>
        </div>
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

      {/* Custom Audio Player */}
      <div
        className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-300 ${isPlaying
          ? "bg-purple-500/10 border border-purple-500/20 shadow-[0_0_15px_rgba(168,85,247,0.1)]"
          : "bg-white/5 border border-white/5 hover:border-white/10"
          }`}
      >
        <button
          onClick={togglePlayback}
          className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-all duration-200 ${isPlaying
            ? "bg-purple-500 text-white shadow-[0_0_12px_rgba(168,85,247,0.5)] hover:bg-purple-400"
            : "bg-white/10 text-white/70 hover:bg-white/20 hover:text-white"
            }`}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4 ml-0.5" />
          )}
        </button>

        <span className="text-xs font-mono text-white/50 w-9 shrink-0 text-right">
          {formatTime(currentTime)}
        </span>

        <div
          ref={progressRef}
          onClick={handleSeek}
          className="flex-1 h-1.5 bg-white/10 rounded-full cursor-pointer group/bar relative overflow-hidden"
        >
          <div
            className={`h-full rounded-full transition-colors duration-200 relative ${isPlaying
              ? "bg-gradient-to-r from-purple-500 to-blue-400"
              : "bg-white/30"
              }`}
            style={{ width: `${progress}%` }}
          >
            {progress > 0 && (
              <div
                className={`absolute right-0 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full transition-opacity duration-200 ${isPlaying
                  ? "bg-white shadow-[0_0_8px_rgba(168,85,247,0.8)] opacity-100"
                  : "bg-white/60 opacity-0 group-hover/bar:opacity-100"
                  }`}
              />
            )}
          </div>
        </div>

        <span className="text-xs font-mono text-white/30 w-9 shrink-0">
          {formatTime(duration)}
        </span>

        <Volume2
          className={`w-3.5 h-3.5 shrink-0 transition-colors ${isPlaying ? "text-purple-400/60" : "text-white/20"
            }`}
        />
      </div>

      {/* Transcription with word highlighting */}
      {hasWords && (
        <div className="ml-14 p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <FileText className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-xs font-medium text-purple-400/80">
              Transcription
            </span>
          </div>
          <p className="text-sm leading-relaxed flex flex-wrap gap-x-1.5 gap-y-1">
            {track.transcription!.words.map((word, i) => {
              const isActive =
                isPlaying &&
                currentTime >= word.start &&
                currentTime < word.end;
              const isPast = isPlaying && currentTime >= word.end;
              return (
                <span
                  key={i}
                  className={`inline-block rounded px-0.5 transition-all duration-150 ${isActive
                    ? "text-white font-semibold scale-110 bg-purple-500/30 shadow-[0_0_12px_rgba(168,85,247,0.4)]"
                    : isPast
                      ? "text-white/70"
                      : "text-white/35"
                    }`}
                  style={{
                    transform: isActive ? "scale(1.1)" : "scale(1)",
                    transition: "all 0.15s ease-out",
                  }}
                >
                  {word.text}
                </span>
              );
            })}
          </p>
        </div>
      )}
      {!hasWords && track.transcription?.text && (
        <div className="ml-14 p-3 rounded-lg bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 mb-1.5">
            <FileText className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-xs font-medium text-purple-400/80">
              Transcription
            </span>
          </div>
          <p className="text-sm text-white/70 leading-relaxed italic">
            &ldquo;{track.transcription.text}&rdquo;
          </p>
        </div>
      )}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showOtherTracks, setShowOtherTracks] = useState(false);
  const [shimmer, setShimmer] = useState(false);
  const samplesRef = useRef<HTMLDivElement>(null);
  const heroRef = useRef<HTMLElement>(null);

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
    accept: { "audio/*": [".wav", ".mp3", ".m4a", ".flac"] },
    maxFiles: 1,
  });

  // Handle "Try it" from a sample card
  const handleTrySample = async (sampleUrl: string) => {
    try {
      const resp = await fetch(sampleUrl);
      const blob = await resp.blob();
      const fileName = sampleUrl.split("/").pop() || "sample.wav";
      const sampleFile = new File([blob], fileName, { type: "audio/wav" });
      setFile(sampleFile);
      setTracks(null);
      setError(null);
      setShowOtherTracks(false);
      // Trigger shimmer on upload zone
      setShimmer(true);
      setTimeout(() => setShimmer(false), 1500);
      // Scroll back to upload section
      heroRef.current?.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      console.error("Failed to load sample:", err);
    }
  };

  const scrollToSamples = () => {
    samplesRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const processAudio = async () => {
    if (!file) return;

    if (file.size > 20 * 1024 * 1024) {
      setError(
        "File too large. Please use a file smaller than 20MB for the demo."
      );
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
          const submitResponse = await axios.post("/api/separate", {
            audio_base64: base64String,
            sample_rate: 16000,
          });

          const jobId = submitResponse.data.id;
          if (!jobId) {
            throw new Error("Failed to start job: No Job ID returned.");
          }

          console.log("Job started:", jobId);

          const pollInterval = 2000;
          const maxRetries = 60;
          let retries = 0;

          const poll = async () => {
            try {
              const statusResponse = await axios.get(
                `/api/status?id=${jobId}`
              );
              const { status, output, error } = statusResponse.data;

              console.log("Job Status:", status);

              if (status === "COMPLETED") {
                if (output && output.separated_tracks) {
                  setTracks(output.separated_tracks);
                  setIsProcessing(false);
                } else {
                  throw new Error(
                    "Job completed but returned no tracks."
                  );
                }
              } else if (status === "FAILED") {
                throw new Error(error || "Job failed on server.");
              } else {
                retries++;
                if (retries >= maxRetries) {
                  throw new Error(
                    "Job timed out (client-side limit reached)."
                  );
                }
                setTimeout(poll, pollInterval);
              }
            } catch (pollErr: any) {
              console.error("Polling Error:", pollErr);
              setError(
                pollErr.message || "Failed to check job status."
              );
              setIsProcessing(false);
            }
          };

          poll();
        } catch (err: any) {
          console.error("Submission error:", err);
          if (err.response?.status === 413) {
            setError("File is too large for the server to process.");
          } else {
            setError(
              err.response?.data?.error ||
              err.message ||
              "Failed to submit audio. Check console for details."
            );
          }
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

  const cappedTracks = tracks?.slice(0, 7) ?? [];
  const mainTracks = cappedTracks.filter(
    (t) => t.transcription?.words && t.transcription.words.length > 0
  );
  const otherTracks = cappedTracks.filter(
    (t) => !t.transcription?.words || t.transcription.words.length === 0
  );

  return (
    <div className="snap-container">
      {/* ── Section 1: Hero + Upload + Results ── */}
      <section ref={heroRef} className="snap-section relative overflow-hidden flex flex-col">
        {/* Background Orbs */}
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl translate-x-1/2 translate-y-1/2" />

        <main className="flex-1 flex flex-col items-center justify-center p-6 sm:p-24 relative z-10 overflow-y-auto">
          <div className="w-full max-w-3xl space-y-12">
            <header className="text-center space-y-4">
              <h1
                onClick={() => {
                  setFile(null);
                  setTracks(null);
                  setError(null);
                  setShowOtherTracks(false);
                  heroRef.current?.scrollIntoView({ behavior: "smooth" });
                }}
                className="text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 cursor-pointer hover:opacity-80 transition-opacity"
              >
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
                  ${shimmer ? "animate-shimmer" : ""}
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

            {/* Error */}
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

                  {otherTracks.length > 0 && (
                    <div className="space-y-3">
                      <button
                        onClick={() =>
                          setShowOtherTracks(!showOtherTracks)
                        }
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
                              <TrackCard
                                key={track.speaker}
                                track={track}
                              />
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  {mainTracks.length === 0 && (
                    <div className="text-center text-white/40 py-4">
                      <p>
                        No clear main voices detected. All tracks are shown
                        below.
                      </p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>

        {/* Scroll Down Hint — only when on the landing page */}
        {!tracks && (
          <button
            onClick={scrollToSamples}
            className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-white/30 hover:text-white/60 transition-colors z-20 animate-bounce"
          >
            <span className="text-xs font-medium tracking-wider uppercase">
              Samples
            </span>
            <ChevronDown className="w-5 h-5" />
          </button>
        )}
      </section>

      {/* ── Section 2: Samples (only on homepage) ── */}
      {!tracks && (
        <section
          ref={samplesRef}
          className="snap-section relative overflow-hidden flex flex-col items-center justify-center p-6 sm:p-24"
        >
          {/* Background Orbs */}
          <div className="absolute top-1/4 right-0 w-80 h-80 bg-blue-600/15 rounded-full blur-3xl translate-x-1/3" />
          <div className="absolute bottom-1/4 left-0 w-80 h-80 bg-purple-600/15 rounded-full blur-3xl -translate-x-1/3" />

          <div className="z-10 w-full max-w-3xl space-y-10">
            <header className="text-center space-y-3">
              <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-400">
                Sample Audio
              </h2>
              <p className="text-base text-white/50">
                Listen to mixed-speaker audio and try separating voices
              </p>
            </header>

            <div className="grid gap-4 sm:grid-cols-3">
              {SAMPLES.map((sample) => (
                <SampleCard
                  key={sample.file}
                  sample={sample}
                  onTry={handleTrySample}
                />
              ))}
            </div>

            {/* Footer */}
            <footer className="pt-8 border-t border-white/10 text-center">
              <a
                href="mailto:sjiradej@ucsc.edu"
                className="inline-flex items-center gap-2 text-sm text-white/40 hover:text-purple-400 transition-colors"
              >
                <Mail className="w-4 h-4" />
                sjiradej@ucsc.edu
              </a>
            </footer>
          </div>
        </section>
      )}
    </div>
  );
}
