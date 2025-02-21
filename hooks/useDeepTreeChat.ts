import { useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

const STORAGE_KEY = '@deep_tree_chat_messages';

export function useDeepTreeChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadMessages();
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      saveMessages(messages);
    }
  }, [messages]);

  const loadMessages = async () => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEY);
      if (stored) {
        setMessages(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Error loading messages:', error);
    }
  };

  const saveMessages = async (msgs: Message[]) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(msgs));
    } catch (error) {
      console.error('Error saving messages:', error);
    }
  };

  const sendMessage = (text: string) => {
    if (!text.trim()) return;

    setIsLoading(true);
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user'
    };
    setMessages(prev => [...prev, newMessage]);
    setIsLoading(false);
  };

  const clearHistory = async () => {
    setMessages([]);
    await AsyncStorage.removeItem(STORAGE_KEY);
  };

  return {
    messages,
    isLoading,
    sendMessage,
    clearHistory
  };
}