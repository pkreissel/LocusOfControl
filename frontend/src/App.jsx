import { useState, useEffect } from 'react'
import './index.css'

// Score used for sorting and the badge, given the active method and sort mode.
function getScore(post, method, sortMode, vectorWeight = 2.5) {
  if (method === 'vector') {
    return (post.heavyRankerScore || 0) + (vectorWeight * (post.vectorInternalScore || 0))
  }
  if (method === 'disagreement') return Math.abs((post.internalScore || 0) - (post.vectorInternalScore || 0))
  return sortMode === 'internal' ? post.internalScore : post.externalScore
}

function PostCard({ post, sortMode, method, vectorWeight }) {
  const [liked, setLiked] = useState(false)
  const [reposted, setReposted] = useState(false)

  const handleLike = () => setLiked(!liked)
  const handleRepost = () => setReposted(!reposted)

  const parts = post.uri.split('/')
  const rkey = parts[parts.length - 1]
  const bskyUrl = `https://bsky.app/profile/${post.did}/post/${rkey}`

  const shortDid = post.did.substring(0, 15) + '...'

  // Determine which score to highlight based on current method and sort mode
  const isVector = method === 'vector'
  const isInternal = isVector || sortMode === 'internal'
  const scoreToDisplay = getScore(post, method, sortMode, vectorWeight)
  const badgeLabel = isVector ? 'Combined Rank' : (isInternal ? 'Internal LoC' : 'External LoC')
  const badgeEmoji = isVector ? '🏅' : (isInternal ? '🧠' : '🌍')

  return (
    <div className="post-card">
      <div className="post-header">
        <div className="did-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {post.avatar ? (
            <img src={post.avatar} alt="avatar" style={{ width: '24px', height: '24px', borderRadius: '50%', objectFit: 'cover' }} />
          ) : (
            <span>👤</span>
          )}
          <span style={{ fontWeight: 'bold' }}>{post.displayName || post.handle || shortDid}</span>
        </div>
        
        {method === 'disagreement' ? (
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <div className="score-badge">🧠 {((post.internalScore || 0) * 100).toFixed(1)}%</div>
            <div className="score-badge">🎯 {((post.vectorInternalScore || 0) * 100).toFixed(1)}%</div>
          </div>
        ) : (
          <div className={`score-badge ${!isInternal ? 'external-badge' : ''}`}>
            {badgeEmoji} {badgeLabel}: {isVector ? (scoreToDisplay || 0).toFixed(2) : ((scoreToDisplay || 0) * 100).toFixed(1) + '%'}
          </div>
        )}
      </div>
      
      <div className="post-content">
        {post.text}
      </div>
      
      <div className="post-actions">
        <button 
          className={`action-btn ${liked ? 'liked' : ''}`}
          onClick={handleLike}
        >
          {liked ? '❤️' : '🤍'} Like
        </button>
        <button 
          className={`action-btn ${reposted ? 'reposted' : ''}`}
          onClick={handleRepost}
        >
          {reposted ? '🔁' : '🔄'} Repost
        </button>
        
        <a href={bskyUrl} target="_blank" rel="noopener noreferrer" className="bsky-link">
          View on Bluesky ↗
        </a>
      </div>
    </div>
  )
}

function App() {
  const [posts, setPosts] = useState([])
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState(null)
  const [sortMode, setSortMode] = useState('internal') // 'internal' or 'external'
  const [method, setMethod] = useState('classifier') // 'classifier' or 'vector'
  const [vectorWeight, setVectorWeight] = useState(2.5) // Default weight for vector score vs heavy ranker

  const fetchPosts = () => {
    setPosts([])
    setScanning(true)
    setError(null)

    // method=both: every post carries classifier and vector scores, so
    // switching methods client-side compares them on the same feed items
    const eventSource = new EventSource(`/api/posts?method=both&vector_weight=${vectorWeight}`)

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.error !== undefined) {
        setError(data.error || 'Unknown server error (see backend logs)')
        eventSource.close()
        setScanning(false)
        return
      }
      // Ignore any message that isn't a scoreable post
      if (!data.uri) return

      // Append new post to the list
      setPosts(prev => {
        // Prevent duplicates
        if (prev.some(p => p.uri === data.uri)) return prev;

        const newList = [...prev, data];
        // Note: the list is re-sorted dynamically during render and state updates,
        // but we keep the base list as is here, or we sort it based on current mode
        return newList.sort((a, b) =>
          (getScore(b, method, sortMode, vectorWeight) || 0) - (getScore(a, method, sortMode, vectorWeight) || 0)
        );
      })
    }
    
    eventSource.addEventListener('end', () => {
      eventSource.close()
      setScanning(false)
    })
    
    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err)
      setError("Connection to server lost.")
      eventSource.close()
      setScanning(false)
    }
    
    return () => {
      eventSource.close()
    }
  }

  // When sort mode, method, or vector weight changes, re-sort the existing posts
  useEffect(() => {
    setPosts(prev => {
      const newList = [...prev];
      return newList.sort((a, b) =>
        (getScore(b, method, sortMode, vectorWeight) || 0) - (getScore(a, method, sortMode, vectorWeight) || 0)
      );
    })
  }, [sortMode, method, vectorWeight])

  // Initial fetch
  useEffect(() => {
    const cleanup = fetchPosts()
    return cleanup
    // eslint-disable-next-line
  }, [])

  return (
    <div className="app-container">
      <header className="header">
        <h1>Discover: Locus of Control</h1>
        <p>Streaming your personalized Discover feed and classifying locus of control.</p>
        
        <div className="sort-controls" style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
          <button
            className={`toggle-btn ${method === 'classifier' ? 'active' : ''}`}
            onClick={() => setMethod('classifier')}
          >
            🤖 Classifier
          </button>
          <button
            className={`toggle-btn ${method === 'vector' ? 'active' : ''}`}
            onClick={() => setMethod('vector')}
          >
            🎯 Vector
          </button>
          <button
            className={`toggle-btn ${method === 'disagreement' ? 'active' : ''}`}
            onClick={() => setMethod('disagreement')}
          >
            ⚖️ Disagreements
          </button>
        </div>

        <div className="sort-controls" style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
          <button
            className={`toggle-btn ${sortMode === 'internal' ? 'active' : ''}`}
            onClick={() => setSortMode('internal')}
          >
            🧠 Sort by Internal
          </button>
            <button
              className={`toggle-btn ${sortMode === 'external' ? 'active' : ''}`}
              onClick={() => setSortMode('external')}
              disabled={method === 'vector'}
              style={method === 'vector' ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
              title={method === 'vector' ? 'The vector method only scores internal LoC' : undefined}
            >
              🌍 Sort by External
            </button>
          </div>

          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
              <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                Vector Score vs Heavy Ranker Weight: <strong>{vectorWeight}</strong>
              </label>
              <input 
                type="range" 
                min="0" 
                max="10" 
                step="0.1" 
                value={vectorWeight} 
                onChange={(e) => setVectorWeight(parseFloat(e.target.value))} 
                style={{ width: '250px' }}
              />
              <button 
                onClick={fetchPosts}
                disabled={scanning}
                style={{
                  marginTop: '0.5rem',
                  padding: '0.4rem 1rem',
                  background: 'var(--accent-color)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: scanning ? 'not-allowed' : 'pointer',
                  opacity: scanning ? 0.7 : 1
                }}
              >
                {scanning ? 'Scanning...' : 'Fetch New Feed'}
              </button>
            </div>
          </div>
        </header>

      <main>
        {error && (
          <div style={{ textAlign: 'center', color: '#ef4444', padding: '2rem', marginBottom: '2rem', background: 'var(--surface-color)', borderRadius: '16px' }}>
            <h2>Error occurred</h2>
            <p>{error}</p>
            <button 
              onClick={fetchPosts}
              style={{
                marginTop: '1rem',
                padding: '0.5rem 1rem',
                background: 'var(--accent-color)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          </div>
        )}

        {method === 'disagreement' && posts.length > 0 && (() => {
          const vecMisses = posts.filter(p => (p.internalScore || 0) >= 0.15 && (p.vectorInternalScore || 0) < 0.25).length;
          const nliMisses = posts.filter(p => (p.internalScore || 0) < 0.15 && (p.vectorInternalScore || 0) >= 0.25).length;
          const bothAgree = posts.filter(p => (p.internalScore || 0) >= 0.15 && (p.vectorInternalScore || 0) >= 0.25).length;
          return (
            <div style={{ background: 'var(--surface-color)', padding: '1rem', borderRadius: '12px', marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ef4444' }}>{vecMisses}</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Vector Misses<br/>(NLI caught)</div>
              </div>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#10b981' }}>{bothAgree}</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Both Agree</div>
              </div>
              <div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#f59e0b' }}>{nliMisses}</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>NLI Misses<br/>(Vector caught)</div>
              </div>
            </div>
          );
        })()}

        <div className="post-list">
          {posts.filter(post => {
            if (method !== 'disagreement') return true;
            const isClassifierInternal = (post.internalScore || 0) >= 0.15;
            const isVectorInternal = (post.vectorInternalScore || 0) >= 0.25;
            return isClassifierInternal !== isVectorInternal;
          }).map((post, i) => (
            <PostCard key={post.uri || i} post={post} sortMode={sortMode} method={method} vectorWeight={vectorWeight} />
          ))}
        </div>
        
        {scanning && (
          <div className="loading-container" style={{ marginTop: posts.length > 0 ? '2rem' : '4rem' }}>
            <div className="spinner"></div>
            <div className="loading-text">Scanning Discover feed...</div>
            <div className="loading-subtext">Classifying posts in real-time.</div>
          </div>
        )}
        
        {!scanning && posts.length === 0 && !error && (
          <p style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
            No prominent locus posts found in this batch. Try scanning again later.
          </p>
        )}
      </main>
    </div>
  )
}

export default App
