var canvas = document.querySelector('canvas')
const c = canvas.getContext('2d')

canvas.width = innerWidth
canvas.height = innerHeight

// c.lineWidth = 100;
// c.filter = 'blur(50px)';
// c.strokeStyle = stroke_color
// c.fillStyle = stroke_color

const wave = {
    y: canvas.height / 2,
    length: 0.01,
    amplitude: 100,
    frequency: 0.01
}

const strokeColor = {
    h: 150,
    s: 50,
    l: 50
}

let inc = wave.frequency

function animate() {
    requestAnimationFrame(animate)
    c.fillStyle = 'rgba(0,0,0,0.001)'
    c.fillRect(0, 0, canvas.width, canvas.height)

    c.beginPath()
    c.moveTo(0, canvas.height / 2)

    for (let i = 0; i < canvas.width; i++) {
        c.lineTo(i, wave.y + Math.sin(i * wave.length + inc) * wave.amplitude)
    }

    c.strokestyle = `hsl(${strokeColor.h}, ${strokeColor.s}%, ${strokeColor.l}%)` // something wrong with this 
    c.stroke()
    inc += wave.frequency
}

animate()