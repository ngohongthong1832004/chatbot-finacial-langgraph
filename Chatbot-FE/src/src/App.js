import React, { useState, useEffect } from 'react';
import './App.css';

// Default users
const DEFAULT_USERS = [
  { username: 'heineken', password: '2' },
  { username: 'tiger', password: '2' },
  { username: '333', password: '2' }
];

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
      
      // Load user sessions
      const userSessions = localStorage.getItem(`sessions_${username}`);
      if (userSessions) {
        const sessions = JSON.parse(userSessions);
        setChatSessions(sessions);
        if (sessions.length > 0) {
          setCurrentSessionId(sessions[sessions.length - 1].id);
          setMessages(sessions[sessions.length - 1].messages);
        } else {
          // Create new session if user has no sessions
          createNewSession();
        }
      } else {
        // Create new session for new user
        createNewSession();
      }
    } else {
      setError('Invalid username or password');
    }
  };

  const handleGoogleLogin = async () => {
    try {
      // Initialize Google Sign-In
      const auth2 = await window.gapi.auth2.getAuthInstance();
      const googleUser = await auth2.signIn();
      const profile = googleUser.getBasicProfile();
      
      // Set user info
      setIsLoggedIn(true);
      setUsername(profile.getName());
      localStorage.setItem('username', profile.getName());
      localStorage.setItem('userEmail', profile.getEmail());
      
      // Create a new session
      createNewSession();
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

  const createNewSession = () => {
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
    localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
  };

  const handleOptionClick = (option) => {
    let message = '';
    switch(option) {
      case 1:
        message = "Tôi muốn được tư vấn về đầu tư";
        break;
      case 2:
        message = "Tôi cần tư vấn về quản lý tài chính";
        break;
      case 3:
        message = "Tôi muốn phân tích thị trường";
        break;
      case 4:
        message = "Tôi cần lập kế hoạch tài chính";
        break;
      default:
        message = "Xin lỗi, tôi không hiểu lựa chọn của bạn";
    }
    
    const newMessage = {
      text: message,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };
    
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    
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

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      const newMessage = {
        text: inputMessage,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString()
      };
      
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);
      
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
      // Save updated sessions for current user
      localStorage.setItem(`sessions_${username}`, JSON.stringify(updatedSessions));
      
      setInputMessage('');
    }
  };

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
            <h3>BIA TƯƠI 3000 Welcome</h3>
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
                <p>{message.text}</p>
                <span className="timestamp">{message.timestamp}</span>
              </div>
            </div>
          ))}
          
          {messages.length === 0 && (
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
        </div>
        
        <div className="input-container">
          <form onSubmit={handleSendMessage} className="input-form">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="message-input"
            />
            <button type="submit" className="send-button">
              <i className="fas fa-paper-plane"></i>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
