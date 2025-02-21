
import type { Plugin } from '../hooks/usePluginManager';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface LlamaConfig {
  modelPath: string;
  contextSize?: number;
  threads?: number;
}

class LlamaIntegration {
  private config: LlamaConfig | null = null;

  async initialize(config: LlamaConfig) {
    this.config = config;
    await AsyncStorage.setItem('llama_config', JSON.stringify(config));
  }

  async generateResponse(prompt: string) {
    if (!this.config) {
      throw new Error('Llama not initialized');
    }
    
    try {
      const response = await fetch('http://0.0.0.0:8080/completion', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt,
          model: this.config.modelPath,
          context_size: this.config.contextSize || 2048,
          threads: this.config.threads || 4
        })
      });
      
      return response.json();
    } catch (error) {
      console.error('Failed to generate response:', error);
      throw error;
    }
  }

  async cleanup() {
    await AsyncStorage.removeItem('llama_config');
    this.config = null;
  }
}

const llamaIntegration = new LlamaIntegration();

export const llamaPlugin: Plugin = {
  id: 'llama-plugin',
  name: 'Llama.cpp Integration',
  version: '1.0.0',
  enabled: false,
  async initialize() {
    const storedConfig = await AsyncStorage.getItem('llama_config');
    if (storedConfig) {
      await llamaIntegration.initialize(JSON.parse(storedConfig));
    }
  },
  cleanup: () => llamaIntegration.cleanup()
};

export default llamaIntegration;
