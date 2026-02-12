
"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { UploadCloud, Music, Play, Pause, Loader2, AlertCircle, Download } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Track {
  speaker: number;
  audio_base64: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      setFile(acceptedFiles[0]);
      setTracks(null);
      setError(null);
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

    try {
      // Convert file to Base64
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = async () => {
        const base64String = (reader.result as string).split(",")[1]; // Remove data URL prefix

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
              err.response?.data?.error || "Failed to process audio. Check console for details."
            );
          }
        } finally {
          setIsProcessing(false);
        }
      };

      reader.onerror = () => {
        console.error("FileReader Error:", reader.error);
        setError(`Failed to read file: ${reader.error?.message || "Unknown error"}`);
        setIsProcessing(false);
      };
    } catch (err) {
      console.error("Unexpected error:", err);
      setError("An unexpected error occurred.");
      setIsProcessing(false);
    }
  };

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
                    : "WAV, MP3, FLAC • Max 10MB"}
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
              className="space-y-4"
            >
              <h2 className="text-2xl font-semibold text-white/80 pl-2 border-l-4 border-purple-500">
                Separated Tracks ({tracks.length})
              </h2>
              <div className="grid gap-4">
                {tracks.map((track, i) => (
                  <div
                    key={i}
                    className="glass p-4 rounded-xl flex items-center justify-between gap-4 group hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-300 font-bold">
                        {track.speaker}
                      </div>
                      <div>
                        <p className="font-medium text-white/90">Speaker {track.speaker}</p>
                        <p className="text-xs text-white/40">Separated Track</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
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
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
