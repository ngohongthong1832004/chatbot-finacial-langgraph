document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const themeBtn = document.getElementById('theme-btn');
    
    // Theme toggle functionality
    themeBtn.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        themeBtn.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Reset height if empty
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });
    
    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const userMessage = userInput.value.trim();
        if (!userMessage) return;
        
        // Add user message to chat
        addMessage(userMessage, 'user');
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Add typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing';
        typingIndicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
        
        try {
            // Send request to API
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: userMessage
                })
            });
            
            // Remove typing indicator
            chatMessages.removeChild(typingIndicator);
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            
            // Add system response to chat
            addMessage(data.answer, 'system');
            
        } catch (error) {
            // Remove typing indicator
            if (typingIndicator.parentNode === chatMessages) {
                chatMessages.removeChild(typingIndicator);
            }
            
            // Add error message
            addMessage('Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau.', 'system');
            console.error('Error:', error);
        }
    });
    
    // Function to add a message to the chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process markdown-like syntax for code blocks
        let processedText = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Process for SQL and tables if it contains them
        if (processedText.includes('SQL Query Results:') || processedText.includes('|')) {
            processedText = processedText.replace(/\|([^|]+)\|/g, '<div class="table-cell">$1</div>');
            processedText = processedText.replace(/\n\|[\-\|]+\|\n/g, '');
        }
        
        messageContent.innerHTML = `<p>${processedText}</p>`;
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        messageTime.textContent = `${hours}:${minutes}`;
        
        messageContent.appendChild(messageTime);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
    }
    
    // Function to scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Initialize by focusing on input
    userInput.focus();
});