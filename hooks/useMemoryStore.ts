import { useState, useEffect } from 'react';
import { MemoryClient } from '../integrations/mem0';

export function useMemoryStore(userId: string) {
  const [memories, setMemories] = useState<string[]>([]);
  const client = new MemoryClient(process.env.EXPO_PUBLIC_MEM0_API_KEY || '');

  useEffect(() => {
    loadMemories();
  }, [userId]);

  const loadMemories = async () => {
    try {
      const response = await client.get({ user_id: userId });
      setMemories(response.memories);
    } catch (error) {
      console.error('Failed to load memories:', error);
    }
  };

  const saveMemory = async (content: string) => {
    try {
      await client.add([{ role: "user", content }], { user_id: userId });
      await loadMemories();
    } catch (error) {
      console.error('Failed to save memory:', error);
    }
  };

  return { memories, saveMemory };
}