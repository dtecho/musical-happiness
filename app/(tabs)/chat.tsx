import { useState, useRef, useEffect } from 'react';
import { View, ScrollView, TextInput, TouchableOpacity } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useDeepTreeChat, type Message } from '@/hooks/useDeepTreeChat';

export default function ChatScreen() {
  const { messages, sendMessage, isLoading, clearHistory } = useDeepTreeChat();
  const [inputText, setInputText] = useState('');
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    if (scrollViewRef.current) {
      scrollViewRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  return (
    <ThemedView style={{ flex: 1 }}>
      <ScrollView
        ref={scrollViewRef}
        style={{ flex: 1, padding: 10 }}
        contentContainerStyle={{ paddingBottom: 20 }}
      >
        {messages.map((message) => (
          <View
            key={message.id}
            style={{
              alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
              backgroundColor: message.sender === 'user' ? '#007AFF' : '#E5E5EA',
              padding: 10,
              borderRadius: 10,
              marginVertical: 5,
              maxWidth: '80%',
            }}
          >
            <ThemedText
              style={{
                color: message.sender === 'user' ? 'white' : 'black',
              }}
            >
              {message.text}
            </ThemedText>
          </View>
        ))}
      </ScrollView>
      <View style={{ padding: 10, flexDirection: 'row', alignItems: 'center' }}>
        <TextInput
          style={{
            flex: 1,
            borderWidth: 1,
            borderColor: '#ccc',
            borderRadius: 20,
            padding: 10,
            marginRight: 10,
          }}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type a message..."
          onSubmitEditing={() => {
            sendMessage(inputText);
            setInputText('');
          }}
        />
        <TouchableOpacity
          onPress={() => {
            sendMessage(inputText);
            setInputText('');
          }}
          disabled={isLoading}
          style={{
            backgroundColor: '#007AFF',
            padding: 10,
            borderRadius: 20,
          }}
        >
          <ThemedText style={{ color: 'white' }}>Send</ThemedText>
        </TouchableOpacity>
      </View>
    </ThemedView>
  );
}