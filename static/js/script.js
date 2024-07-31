const canvas = document.querySelector('canvas')
const c = canvas.getContext('2d')

canvas.width = innerWidth
canvas.height = innerHeight

const wave = {
  y: canvas.height / 2,
  length: 0.01,
  amplitude: canvas.height / 3,
  frequency: 0.01
}

const strokeColor = {
  h: 258,
  s: 97,
  l: 60,
  a: 0.2
}

var path1 = new Path2D()
path1.rect(0, 0, canvas.width, canvas.height)

var gradient = c.createLinearGradient(0, 0, canvas.width, canvas.height)
gradient.addColorStop(0, '#4f4080')
gradient.addColorStop(0.4, '#754ab3')
gradient.addColorStop(0.6, '#9553ff')
gradient.addColorStop(1, '#9553ff')


// TODO: set color of stroke to rgba (so can use alpha) 
let increment = wave.frequency

function animate() {
  requestAnimationFrame(animate)
//   c.fillStyle = `rgba(${backgroundColor.r}, ${backgroundColor.g}, ${backgroundColor.b}, ${backgroundColor.a})`
//   c.fillRect(0, 0, canvas.width, canvas.height)

  c.fillStyle = gradient
  c.fill(path1)

  c.beginPath()
  c.moveTo(0, canvas.height / 2)

  c.lineWidth = 10;
  c.shadowColor = `hsl(${strokeColor.h}, ${strokeColor.s}%, ${Math.abs(strokeColor.l * Math.sin(increment))}%)`;
  c.shadowBlur = 50;

  for (let i = 0; i < canvas.width; i++) {
    c.lineTo(i, wave.y + Math.sin(i * wave.length + increment) * wave.amplitude * Math.sin(increment))
  }

  c.strokeStyle = `hsl(${strokeColor.h}, ${Math.abs(strokeColor.s * Math.sin(increment))}%, ${Math.abs(strokeColor.l * Math.sin(increment))}%)`
  c.stroke()
  increment += wave.frequency
}

animate()