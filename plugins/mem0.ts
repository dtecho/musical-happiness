
import type { Plugin } from '../hooks/usePluginManager';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface Mem0Config {
  apiKey: string;
}

class Mem0Integration {
  private config: Mem0Config | null = null;

  async initialize(config: Mem0Config) {
    this.config = config;
    await AsyncStorage.setItem('mem0_config', JSON.stringify(config));
  }

  async addMemory(messages: Array<{role: string, content: string}>, userId: string) {
    if (!this.config) {
      throw new Error('Mem0 not initialized');
    }
    
    try {
      const response = await fetch('https://api.mem0.ai/memories', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`
        },
        body: JSON.stringify({
          messages,
          user_id: userId
        })
      });
      
      return response.json();
    } catch (error) {
      console.error('Failed to add memory:', error);
      throw error;
    }
  }

  async cleanup() {
    await AsyncStorage.removeItem('mem0_config');
    this.config = null;
  }
}

class Mem0Plugin implements Plugin {
  id = 'mem0-plugin';
  name = 'Mem0 Integration';
  version = '1.0.0';
  enabled = false;
  private integration = new Mem0Integration();

  async initialize() {
    const storedConfig = await AsyncStorage.getItem('mem0_config');
    if (storedConfig) {
      await this.integration.initialize(JSON.parse(storedConfig));
    }
  }

  cleanup() {
    return this.integration.cleanup();
  }
}

export const mem0Plugin = new Mem0Plugin();
export default mem0Plugin;
