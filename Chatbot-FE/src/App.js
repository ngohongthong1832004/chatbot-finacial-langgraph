import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

// Default users
const DEFAULT_USERS = [
  { username: 'heineken', password: '2' },
  { username: 'tiger', password: '2' },
  { username: '333', password: '2' }
];

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem('isLoggedIn') === 'true';
  });
  const [username, setUsername] = useState(() => {
    return localStorage.getItem('username') || '';
  });
  const [password, setPassword] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingSessionTitle, setEditingSessionTitle] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Load sessions when component mounts or username changes
  useEffect(() => {
    if (isLoggedIn && username) {
      const userSessions = localStorage.getItem(`sessions_${username}`);
      if (userSessions) {
        const sessions = JSON.parse(userSessions);
        setChatSessions(sessions);
        if (sessions.length > 0) {
          setCurrentSessionId(sessions[sessions.length - 1].id);
          setMessages(sessions[sessions.length - 1].messages);
        } else {
          createNewSession();
        }
      } else {
        createNewSession();
      }
    }
  }, [isLoggedIn, username]);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const usernameFromGoogle = params.get("username");
    const emailFromGoogle = params.get("email");

    if (usernameFromGoogle && emailFromGoogle) {
      // Lưu thông tin login
      setUsername(usernameFromGoogle);
      setIsLoggedIn(true);
      localStorage.setItem("username", usernameFromGoogle);
      localStorage.setItem("userEmail", emailFromGoogle);
      localStorage.setItem("isLoggedIn", "true");

      // ✅ Không gọi createNewSession ở đây nữa

      // Xoá query khỏi URL
      window.history.replaceState({}, document.title, "/");
    }
  }, []);

  const handleLogin = (e) => {
    e.preventDefault();
    setError('');
    
    // Check against DEFAULT_USERS directly
    const user = DEFAULT_USERS.find(u => u.username === username && u.password === password);
    
    if (user) {
      setIsLoggedIn(true);
      // Save login info to localStorage
      localStorage.setItem('isLoggedIn', 'true');
      localStorage.setItem('username', username);
      localStorage.setItem('lastLogin', new Date().toISOString());
      
      // Sessions will be loaded by useEffect
    } else {
      setError('Invalid username or password');
    }
  };

  // const handleGoogleLogin = async () => {
  const handleGoogleLogin =  () => {
    try {
    //   // Initialize Google Sign-In
    //   const auth2 = await window.gapi.auth2.getAuthInstance();
    //   const googleUser = await auth2.signIn();
    //   const profile = googleUser.getBasicProfile();
      
    //   // Set user info
    //   setIsLoggedIn(true);
    //   setUsername(profile.getName());
    //   localStorage.setItem('username', profile.getName());
    //   localStorage.setItem('userEmail', profile.getEmail());
      
    //   // Create a new session
    //   createNewSession();
      window.location.href = "http://localhost:8000/api/login";
    } catch (error) {
      console.error('Google login error:', error);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
    setMessages([]);
    setCurrentSessionId(null);
    
    // Remove login info from localStorage
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('username');
    localStorage.removeItem('lastLogin');
    localStorage.removeItem('userEmail');
  };

  // const createNewSession = () => {
  //   const newSession = {
  //     id: Date.now(),
  //     title: `Chat ${chatSessions.length + 1}`,
  //     messages: [],
  //     createdAt: new Date().toISOString()
  //   };
  //   const updatedSessions = [...chatSessions, newSession];
  //   setChatSessions(updatedSessions);
  //   setCurrentSessionId(newSession.id);
  //   setMessages([]);
  //   localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
  // };
  const createNewSession = (customUsername) => {
    const finalUsername = customUsername || username;
    const newSession = {
      id: Date.now(),
      title: `Chat ${chatSessions.length + 1}`,
      messages: [],
      createdAt: new Date().toISOString()
    };
    const updatedSessions = [...chatSessions, newSession];
    setChatSessions(updatedSessions);
    setCurrentSessionId(newSession.id);
    setMessages([]);
    localStorage.setItem(`sessions_${finalUsername}`, JSON.stringify(updatedSessions));
  };


  const handleOptionClick = async (option) => {
    let message = '';
    let botResponse = '';
    const active = true; // Set this to false to use predefined responses
    
    switch(option) {
      case 1:
        message = "Tôi muốn được tư vấn về đầu tư";
        botResponse = `# Tư vấn đầu tư

Tôi có thể giúp bạn phân tích và đưa ra các gợi ý đầu tư phù hợp. Dưới đây là một số lĩnh vực đầu tư phổ biến:

## 1. Chứng khoán
- Cổ phiếu
- Trái phiếu
- ETF

## 2. Bất động sản
- Nhà ở
- Đất nền
- Căn hộ cho thuê

## 3. Tiền điện tử
\`\`\`javascript
// Ví dụ về phân tích xu hướng
const analyzeTrend = (data) => {
  const sma = calculateSMA(data, 20);
  const ema = calculateEMA(data, 20);
  return { sma, ema };
};
\`\`\`

Bạn muốn tìm hiểu thêm về lĩnh vực nào?`;
        break;
      case 2:
        message = "Tôi cần tư vấn về quản lý tài chính";
        botResponse = `# Quản lý tài chính cá nhân

## Các nguyên tắc cơ bản:
1. **Chi tiêu thông minh**
2. **Tiết kiệm đều đặn**
3. **Đầu tư dài hạn**

## Công thức tính lãi kép:
\`\`\`python
def compound_interest(principal, rate, time):
    amount = principal * (1 + rate/100) ** time
    return amount

# Ví dụ
principal = 1000000  # 1 triệu
rate = 8  # 8% mỗi năm
time = 10  # 10 năm
\`\`\`

Bạn cần tư vấn cụ thể về vấn đề nào?`;
        break;
      case 3:
        message = "Tôi muốn phân tích thị trường";
        botResponse = `# 📊 Phân tích thị trường

Phân tích thị trường là quá trình đánh giá tình hình kinh doanh của các doanh nghiệp niêm yết thông qua **dữ liệu tài chính** và **các chỉ số định lượng**. Dưới đây là một số chỉ số quan trọng:

## 🔍 Chỉ số phổ biến

- **P/E Ratio**: Tỷ lệ giá trên lợi nhuận
- **ROE**: Tỷ suất sinh lời trên vốn chủ sở hữu
- **EPS**: Lợi nhuận trên mỗi cổ phiếu
- **Market Cap**: Vốn hóa thị trường

## 🧮 Ví dụ về công thức tính:

\`\`\`python
def pe_ratio(price, eps):
    return price / eps if eps != 0 else None

def roe(net_income, equity):
    return (net_income / equity) * 100
\`\`\`

## 📋 Dữ liệu thị trường mẫu:

| Symbol | Company             | Sector       | Market Cap (Tỷ USD) | P/E Ratio | EPS  | ROE (%) |
|--------|---------------------|--------------|---------------------:|-----------:|------:|--------:|
| AAPL   | Apple Inc.          | Technology   |             3143.8  |      33.22 |  5.0 |    28.7 |
| MSFT   | Microsoft Corp.     | Technology   |             2800.5  |      34.10 |  7.5 |    43.1 |
| NVDA   | NVIDIA Corporation  | Semiconduct. |             1200.4  |      72.00 | 12.5 |    58.2 |
| JPM    | JPMorgan Chase      | Finance      |              490.0  |      10.50 | 11.3 |    16.4 |
| XOM    | Exxon Mobil Corp    | Energy       |              420.3  |       9.40 | 10.1 |    24.6 |

## 📈 Biểu đồ xu hướng (ý tưởng):
- Bạn có thể vẽ biểu đồ P/E theo thời gian
- Hoặc dùng kỹ thuật RSI để đánh giá điểm mua-bán

\`\`\`typescript
function calculateRSI(data: number[]): number {
  const gains = data.filter(d => d > 0).length;
  const losses = data.filter(d => d < 0).length;
  const rs = gains / (losses || 1);
  return 100 - (100 / (1 + rs));
}
\`\`\`

Bạn muốn phân tích thêm công ty hoặc ngành nào cụ thể?`;
        break;
      case 4:
        message = "Tôi cần lập kế hoạch tài chính";
        botResponse = `# Lập kế hoạch tài chính

## Các bước cơ bản:
1. Xác định mục tiêu
2. Đánh giá tình hình hiện tại
3. Lập kế hoạch chi tiết
4. Theo dõi và điều chỉnh

## Công cụ tính toán:
\`\`\`excel
=PMT(rate/12, nper, pv, [fv], [type])
\`\`\`

> Lưu ý: Kế hoạch tài chính cần được điều chỉnh theo tình hình thực tế và mục tiêu cá nhân.

Bạn muốn lập kế hoạch cho mục tiêu nào?`;
        break;
      default:
        message = "Xin lỗi, tôi không hiểu lựa chọn của bạn";
        botResponse = "Vui lòng chọn một trong các tùy chọn trên.";
    }
    
    const newMessage = {
      text: message,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };
    
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setIsLoading(true);
    
    const updatedSessions = chatSessions.map(session => {
      if (session.id === currentSessionId) {
        return {
          ...session,
          messages: updatedMessages
        };
      }
      return session;
    });
    
    setChatSessions(updatedSessions);
    localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    let formattedResponse;
    if (active) {
      // Call API if active is true
      const data = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: message }),
      });

      if (!data.ok) {
        throw new Error('Network response was not ok');
      }

      const responseData = await data.json();
      formattedResponse = responseData.answer;
    } else {
      // Use predefined response if active is false
      formattedResponse = botResponse;
    }
    
    // Format the response with markdown
    // Split the response into sections
    const sections = formattedResponse.split('\n\n');
    
    // Process each section
    formattedResponse = sections.map(section => {
      // If section contains a table
      if (section.includes('|') && section.includes('---')) {
        // Split into lines
        const lines = section.split('\n');
        // Process each line
        return lines.map(line => {
          // If it's a table header separator line, keep it as is
          if (line.includes('|:---|')) {
            return line;
          }
          // If it's a table line, ensure proper spacing
          if (line.includes('|')) {
            return line.trim();
          }
          return line;
        }).join('\n');
      }
      // For non-table sections, add extra newlines
      return section + '\n\n';
    }).join('\n');

    // Add bot response
    const botMessage = {
      text: formattedResponse,
      sender: 'bot',
      timestamp: new Date().toLocaleTimeString()
    };
    
    const finalMessages = [...updatedMessages, botMessage];
    setMessages(finalMessages);
    
    // Update sessions with bot response
    const finalSessions = chatSessions.map(session => {
      if (session.id === currentSessionId) {
        return {
          ...session,
          messages: finalMessages
        };
      }
      return session;
    });
    
    setChatSessions(finalSessions);
    localStorage.setItem(`sessions_${username}`, JSON.stringify(finalSessions));
    setIsLoading(false);
  };

  const deleteSession = (sessionId, e) => {
    e.stopPropagation();
    const updatedSessions = chatSessions.filter(session => session.id !== sessionId);
    setChatSessions(updatedSessions);
    // Save updated sessions for current user
    localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
    
    if (sessionId === currentSessionId) {
      if (updatedSessions.length > 0) {
        setCurrentSessionId(updatedSessions[updatedSessions.length - 1].id);
        setMessages(updatedSessions[updatedSessions.length - 1].messages);
      } else {
        setCurrentSessionId(null);
        setMessages([]);
      }
    }
  };

  const handleRenameSession = (sessionId, e) => {
    e.stopPropagation();
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setEditingSessionId(sessionId);
      setEditingSessionTitle(session.title);
    }
  };

  const saveSessionTitle = (e) => {
    e.preventDefault();
    if (editingSessionTitle.trim()) {
      const updatedSessions = chatSessions.map(session => {
        if (session.id === editingSessionId) {
          return {
            ...session,
            title: editingSessionTitle.trim()
          };
        }
        return session;
      });
      setChatSessions(updatedSessions);
      // Save updated sessions for current user only
      localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
      setEditingSessionId(null);
      setEditingSessionTitle('');
    }
  };

  const switchSession = (sessionId) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setMessages(session.messages);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      const newMessage = {
        text: inputMessage,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString()
      };
      
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);
      setInputMessage('');
      setIsLoading(true);
      
      // Update the current session in chatSessions
      const updatedSessions = chatSessions.map(session => {
        if (session.id === currentSessionId) {
          return {
            ...session,
            messages: updatedMessages
          };
        }
        return session;
      });
      
      setChatSessions(updatedSessions);
      localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
      
      try {
        // Call API with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout

        const response = await fetch(`${API_URL}/api/ask`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: inputMessage }),
          signal: controller.signal
        });

        clearTimeout(timeoutId); // Clear timeout if request completes

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Format the response with markdown
        let formattedResponse = data.answer;
        
        // Split the response into sections
        const sections = formattedResponse.split('\n\n');
        
        // Process each section
        formattedResponse = sections.map(section => {
          // If section contains a table
          if (section.includes('|') && section.includes('---')) {
            // Split into lines
            const lines = section.split('\n');
            // Process each line
            return lines.map(line => {
              // If it's a table header separator line, keep it as is
              if (line.includes('|:---|')) {
                return line;
              }
              // If it's a table line, ensure proper spacing
              if (line.includes('|')) {
                return line.trim();
              }
              return line;
            }).join('\n');
          }
          // For non-table sections, add extra newlines
          return section + '\n\n';
        }).join('\n');

        // Add bot response
        const botMessage = {
          text: formattedResponse,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        
        const finalMessages = [...updatedMessages, botMessage];
        setMessages(finalMessages);
        
        // Update sessions with bot response
        const finalSessions = chatSessions.map(session => {
          if (session.id === currentSessionId) {
            return {
              ...session,
              messages: finalMessages
            };
          }
          return session;
        });
        
        setChatSessions(finalSessions);
        localStorage.setItem(`sessions_${username}`, JSON.stringify(finalSessions));
      } catch (error) {
        console.error('Error:', error);
        // Add error message
        const errorMessage = {
          text: error.name === 'AbortError' 
            ? "Yêu cầu của bạn đã hết thời gian chờ (30 giây). Vui lòng thử lại sau."
            : "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn.",
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString()
        };
        
        const finalMessages = [...updatedMessages, errorMessage];
        setMessages(finalMessages);
        
        // Update sessions with error message
        const finalSessions = chatSessions.map(session => {
          if (session.id === currentSessionId) {
            return {
              ...session,
              messages: finalMessages
            };
          }
          return session;
        });
        
        setChatSessions(finalSessions);
        localStorage.setItem(`sessions_${username}`, JSON.stringify(finalSessions));
      } finally {
        setIsLoading(false);
      }
    }
  };

  const renderMessage = (message) => {
    if (message.sender === 'bot') {
      return (

<ReactMarkdown
  remarkPlugins={[remarkGfm]}
  components={{
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  }}
>
  {message.text}
</ReactMarkdown>
      );
    }
    return <p>{message.text}</p>;
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (!isLoggedIn) {
    return (
      <div className="login-container">
        <div className="login-box">
          <div className="app-logo">
            <i className="fas fa-comments"></i>
          </div>
          <h2>ChatBot Agent</h2>
          {error && <div className="error-message">{error}</div>}
          <form onSubmit={handleLogin}>
            <div className="form-group">
              <i className="fas fa-user"></i>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <i className="fas fa-lock"></i>
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button type="submit" className="login-button">
              <i className="fas fa-sign-in-alt"></i>
              Login
            </button>
          </form>
          <div className="divider">
            <span>or</span>
          </div>
          <button onClick={handleGoogleLogin} className="google-login-button">
            <img
              src="https://www.google.com/favicon.ico"
              alt="Google"
              className="google-icon"
            />
            Continue with Google
          </button>
          {/* <div className="default-users">
            <h4>Default Users:</h4>
            <ul>
              {DEFAULT_USERS.map((user, index) => (
                <li key={index}>
                  Username: {user.username} | Password: {user.password}
                </li>
              ))}
            </ul>
          </div> */}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sessions-section">
          <div className="sessions-header">
            <h4>Chat Sessions</h4>
            <button onClick={createNewSession} className="new-chat-button">
              <i className="fas fa-plus"></i>
              New Chat
            </button>
          </div>
          <div className="sessions-list">
          {chatSessions.map((session) => {
  const isEditing = editingSessionId === session.id;

  return (
    <div
      key={session.id}
      className={`session-item ${session.id === currentSessionId ? 'active' : ''}`}
      onClick={() => switchSession(session.id)}
    >
      {isEditing ? (
        <form onSubmit={saveSessionTitle} className="session-edit-form">
          <input
            type="text"
            value={editingSessionTitle}
            onChange={(e) => setEditingSessionTitle(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            className="session-title-input"
            autoFocus
          />
          <div className="session-actions">
            <button type="submit" className="save-title-button" title="Save">
              <i className="fas  fa-check"></i>
            </button>
            <button
              type="button"
              className="cancel-title-button"
              onClick={(e) => {
                e.stopPropagation();
                setEditingSessionId(null);
                setEditingSessionTitle('');
              }}
              title="Cancel"
            >
              <i className="fas fa-times"></i>
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className="session-content">
            <div className="session-info">
              <span className="session-title">{session.title}</span>
              <span className="session-time">
                {new Date(session.createdAt).toLocaleDateString()}
              </span>
            </div>
          </div>
          <div className="session-actions">
            <button
              className="rename-session-button"
              onClick={(e) => handleRenameSession(session.id, e)}
              title="Rename session"
            >
              <i className="fas fa-pen"></i>
            </button>
            <button
              className="delete-session-button"
              onClick={(e) => deleteSession(session.id, e)}
              title="Delete session"
            >
              <i className="fas fa-trash"></i>
            </button>
          </div>
        </>
      )}
    </div>
  );
})}

          </div>
        </div>
        <div className="user-info">
          <div className="user-profile">
            <h3> Welcome BIA TƯƠI 3000</h3>
          </div>
          <div className='user-info-text-wrap'>
            <div className="user-info-text"> 
              <i className="fas fa-user-circle" style={{ fontSize: '24px', color: 'gray' }}></i>
              <h3>{username}</h3>
            </div>
          <button onClick={handleLogout} className="logout-button">
            <i className="fas fa-sign-out-alt"></i>
            Logout
          </button>
          </div>
        </div>
      </div>
      
      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                {renderMessage(message)}
                <span className="timestamp">{message.timestamp}</span>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot-message">
              <div className="message-content">
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          {messages.length === 0 && !isLoading && (
            <div className="welcome-container">
              <div className="welcome-content">
                <div className="welcome-header">
                  <i className="fas fa-robot"></i>
                  <h2>AI Chatbot LangGraph Agent</h2>
                </div>
                <p className="welcome-text">
                  Tôi là trợ lý AI chuyên về tài chính. Tôi có thể giúp bạn với các vấn đề sau:
                </p>
                <div className="options-grid">
                  <button onClick={() => handleOptionClick(1)} className="option-button">
                    <i className="fas fa-chart-line"></i>
                    Tư vấn đầu tư
                  </button>
                  <button onClick={() => handleOptionClick(2)} className="option-button">
                    <i className="fas fa-wallet"></i>
                    Quản lý tài chính
                  </button>
                  <button onClick={() => handleOptionClick(3)} className="option-button">
                    <i className="fas fa-chart-pie"></i>
                    Phân tích thị trường
                  </button>
                  <button onClick={() => handleOptionClick(4)} className="option-button">
                    <i className="fas fa-calendar-check"></i>
                    Kế hoạch tài chính
                  </button>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="input-container">
          <form onSubmit={handleSendMessage} className="input-form">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="message-input"
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={isLoading}>
              <i className="fas fa-paper-plane"></i>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
