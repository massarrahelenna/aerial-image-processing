document.addEventListener('DOMContentLoaded', function() {
      const video = videojs('tactical-video')
      
      // Canvas setup
      const canvas = document.getElementById('drawingCanvas')
      const ctx = canvas.getContext('2d')
      let isDrawing = false
      let currentTool = 'none'
      
      // Resize canvas to video dimensions
      function resizeCanvas() {
        const videoElement = document.querySelector('.video-js')
        canvas.width = videoElement.offsetWidth
        canvas.height = videoElement.offsetHeight
      }
      
      resizeCanvas()
      window.addEventListener('resize', resizeCanvas)
      
      // Tool buttons functionality
      document.getElementById('drawBtn').addEventListener('click', function() {
        currentTool = 'draw'
        updateToolButtons()
      })
      
      document.getElementById('arrowBtn').addEventListener('click', function() {
        currentTool = 'arrow'
        updateToolButtons()
      })
      
      document.getElementById('lineBtn').addEventListener('click', function() {
        currentTool = 'line'
        updateToolButtons()
      })
      
      document.getElementById('circleBtn').addEventListener('click', function() {
        currentTool = 'circle'
        updateToolButtons()
      })
      
      document.getElementById('textBtn').addEventListener('click', function() {
        currentTool = 'text'
        updateToolButtons()
        showTextInput()
      })
      
      function updateToolButtons() {
        const tools = ['drawBtn', 'arrowBtn', 'lineBtn', 'circleBtn', 'textBtn']
        tools.forEach(tool => {
          const btn = document.getElementById(tool)
          btn.classList.remove('active')
        })
        
        if (currentTool !== 'none') {
          document.getElementById(currentTool + 'Btn').classList.add('active')
        }
      }
      
      // Canvas drawing functionality
      let startX, startY
      
      canvas.addEventListener('mousedown', function(e) {
        if (currentTool === 'none') return
        
        isDrawing = true
        startX = e.offsetX
        startY = e.offsetY
        
        if (currentTool === 'draw') {
          ctx.beginPath()
          ctx.moveTo(startX, startY)
        }
      })
      
      canvas.addEventListener('mousemove', function(e) {
        if (!isDrawing) return
        
        const x = e.offsetX
        const y = e.offsetY
        
        ctx.strokeStyle = '#e74c3c'
        ctx.lineWidth = 3
        ctx.lineCap = 'round'
        
        switch(currentTool) {
          case 'draw':
            ctx.lineTo(x, y)
            ctx.stroke()
            break
          case 'line':
            clearCanvas()
            ctx.beginPath()
            ctx.moveTo(startX, startY)
            ctx.lineTo(x, y)
            ctx.stroke()
            break
          case 'arrow':
            drawArrow(startX, startY, x, y)
            break
          case 'circle':
            const radius = Math.sqrt(Math.pow(x - startX, 2) + Math.pow(y - startY, 2))
            clearCanvas()
            ctx.beginPath()
            ctx.arc(startX, startY, radius, 0, Math.PI * 2)
            ctx.stroke()
            break
        }
      })
      
      canvas.addEventListener('mouseup', function() {
        if (!isDrawing) return
        
        isDrawing = false
        if (currentTool === 'draw') {
          ctx.closePath()
        }
      })
      
      function drawArrow(fromX, fromY, toX, toY) {
        clearCanvas()
        
        // Draw line
        ctx.beginPath()
        ctx.moveTo(fromX, fromY)
        ctx.lineTo(toX, toY)
        ctx.stroke()
        
        // Arrow head
        const headLength = 15
        const angle = Math.atan2(toY - fromY, toX - fromX)
        
        ctx.beginPath()
        ctx.moveTo(toX, toY)
        ctx.lineTo(
          toX - headLength * Math.cos(angle - Math.PI / 6),
          toY - headLength * Math.sin(angle - Math.PI / 6)
        )
        ctx.stroke()
        
        ctx.beginPath()
        ctx.moveTo(toX, toY)
        ctx.lineTo(
          toX - headLength * Math.cos(angle + Math.PI / 6),
          toY - headLength * Math.sin(angle + Math.PI / 6)
        )
        ctx.stroke()
      }
      
      function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      
      function showTextInput() {
        const modal = document.getElementById('annotationModal')
        modal.style.display = 'block'
        modal.style.left = (canvas.offsetLeft + canvas.width / 4) + 'px'
        modal.style.top = (canvas.offsetTop + canvas.height / 3) + 'px'
        
        // Close modal when clicking outside
        document.addEventListener('click', function closeModal(e) {
          if (!modal.contains(e.target) && e.target !== document.getElementById('textBtn')) {
            modal.style.display = 'none'
            document.removeEventListener('click', closeModal)
          }
        })
      }
      
      // Video controls
      document.getElementById('playBtn').addEventListener('click', function() {
        if (video.paused()) {
          video.play()
        } else {
          video.pause()
        }
      })
      
      document.getElementById('rewindBtn').addEventListener('click', function() {
        video.currentTime(Math.max(0, video.currentTime() - 10))
      })
      
      document.getElementById('forwardBtn').addEventListener('click', function() {
        video.currentTime(Math.min(video.duration(), video.currentTime() + 10))
      })
      
      document.getElementById('speedSelect').addEventListener('change', function() {
        video.playbackRate(this.value)
      })
      
      // Timeline functionality
      const timelineBar = document.getElementById('timelineBar')
      const timelineProgress = document.getElementById('timelineProgress')
      const timelineMarkers = document.getElementById('timelineMarkers')
      
      video.on('timeupdate', function() {
        const percent = (video.currentTime() / video.duration()) * 100
        timelineProgress.style.width = percent + '%'
      })
      
      timelineBar.addEventListener('click', function(e) {
        const rect = this.getBoundingClientRect()
        const percent = (e.clientX - rect.left) / rect.width
        video.currentTime(percent * video.duration())
      })
      
      document.getElementById('addMarkerBtn').addEventListener('click', function() {
        const percent = (video.currentTime() / video.duration()) * 100
        const marker = document.createElement('div')
        marker.className = 'marker'
        marker.style.left = percent + '%'
        marker.dataset.type = 'Marcador'
        
        marker.addEventListener('click', function(e) {
          e.stopPropagation()
          video.currentTime((parseFloat(this.style.left) / 100) * video.duration())
        })
        
        timelineMarkers.appendChild(marker)
      })
      
      // Formation player selection
      document.querySelectorAll('.formation-player').forEach(player => {
        player.addEventListener('click', function() {
          if (this.textContent) { // Only if has number
            document.getElementById('playerName').textContent = `Jogador ${this.textContent}`
            document.getElementById('playerPosition').textContent = this.parentElement === document.querySelector('.formation-grid') ? 
              'Atacante' : 'Meio-campo'
            document.getElementById('playerDistance').textContent = `${Math.round(Math.random() * 10)}.${Math.round(Math.random() * 9)} km`
            document.getElementById('playerPasses').textContent = 
              `${Math.round(Math.random() * 30)}/${Math.round(Math.random() * 40)} (${Math.round(Math.random() * 100)}%)`
          }
        })
      })
      
      // Export functionality
      document.getElementById('exportBtn').addEventListener('click', function() {
        alert('Exportando análise tática...')
      })
      
      // Session save
      document.getElementById('saveSessionBtn').addEventListener('click', function() {
        alert('Sessão salva com sucesso!')
      })
    })