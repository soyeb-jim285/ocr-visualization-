"use client";

import { useState, useEffect } from "react";

const scripts = ["English", "বাংলা", "123৪৫৬", "সংযুক্ত"];

function TypingAnimation() {
  const [idx, setIdx] = useState(0);
  const [displayed, setDisplayed] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const word = scripts[idx];
    let timer: ReturnType<typeof setTimeout>;

    if (!isDeleting && displayed.length < word.length) {
      timer = setTimeout(() => setDisplayed(word.slice(0, displayed.length + 1)), 90);
    } else if (!isDeleting && displayed.length === word.length) {
      timer = setTimeout(() => setIsDeleting(true), 1400);
    } else if (isDeleting && displayed.length > 0) {
      timer = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 50);
    } else {
      setIsDeleting(false);
      setIdx((i) => (i + 1) % scripts.length);
    }

    return () => clearTimeout(timer);
  }, [displayed, isDeleting, idx]);

  return (
    <div className="h-10 sm:h-12">
      <span className="font-mono text-xl text-accent-primary sm:text-2xl md:text-3xl">
        {displayed}
        <span className="animate-pulse">|</span>
      </span>
    </div>
  );
}

export function HeroHeader() {
  return (
    <div className="flex flex-col items-center gap-1">
      <h1 className="text-center text-2xl font-semibold tracking-tight text-foreground sm:text-4xl md:text-[2.7rem]">
        Peel Back the Layers of Recognition
      </h1>
      <TypingAnimation />
    </div>
  );
}
