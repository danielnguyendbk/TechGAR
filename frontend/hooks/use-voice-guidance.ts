'use client';

import { useEffect, useRef } from 'react';

export function useVoiceGuidance(message: string | null): void {
  const spoken = useRef(new Set<string>());
  useEffect(() => {
    if (!message || spoken.current.has(message) || typeof window === 'undefined') return;
    const synthesis = window.speechSynthesis;
    if (!synthesis) return;
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'vi-VN';
    synthesis.speak(utterance);
    spoken.current.add(message);
  }, [message]);
}

