// Navbar scroll effect
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 50) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
});

// Theme toggle
const themeToggle = document.getElementById('theme-toggle');
let isDarkTheme = true;

themeToggle.addEventListener('click', () => {
  document.body.style.backgroundColor = isDarkTheme ? '#fff' : '#111';
  document.body.style.color = isDarkTheme ? '#000' : '#fff';
  themeToggle.textContent = isDarkTheme ? 'Light Mode' : 'Dark Mode';
  isDarkTheme = !isDarkTheme;
});

// Smooth scrolling to #second_main
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();

    const targetId = this.getAttribute('href').substring(1);
    const targetElement = document.getElementById(targetId);

    if (targetElement) {
      window.scrollTo({
        top: targetElement.offsetTop,
        behavior: 'smooth'
      });
    }
  });
});

document.addEventListener('DOMContentLoaded', () => {
  // Select all anchor links with hashes
  const links = document.querySelectorAll('a[href*="#"]');
  
  links.forEach(link => {
    link.addEventListener('click', event => {
      // Prevent default behavior
      event.preventDefault();

      // Get the full href including Flask's dynamic URL
      const fullHref = link.getAttribute('href');
      const [baseURL, hash] = fullHref.split('#'); // Split Flask route and hash

      if (hash) {
        const targetElement = document.getElementById(hash);

        if (targetElement) {
          // Scroll smoothly to the target element
          targetElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
          
          // Update the browser's URL without reloading
          history.pushState(null, '', `${baseURL}#${hash}`);
          
          // Delay the reload to ensure smooth scrolling completes
          setTimeout(() => {
            location.reload(true);
          }, 500); // Adjust the timeout duration as needed
        }
      }
    });
  });
});

// function([string1, string2],target id,[color1,color2])    
consoleText(['We are Data Divers', 'VisioCam', 'Made with Love.'], 'text',['tomato','rebeccapurple','lightblue']);

function consoleText(words, id, colors) {
  if (colors === undefined) colors = ['#fff'];
  var visible = true;
  var con = document.getElementById('console');
  var letterCount = 1;
  var x = 1;
  var waiting = false;
  var target = document.getElementById(id)
  target.setAttribute('style', 'color:' + colors[0])
  window.setInterval(function() {

    if (letterCount === 0 && waiting === false) {
      waiting = true;
      target.innerHTML = words[0].substring(0, letterCount)
      window.setTimeout(function() {
        var usedColor = colors.shift();
        colors.push(usedColor);
        var usedWord = words.shift();
        words.push(usedWord);
        x = 1;
        target.setAttribute('style', 'color:' + colors[0])
        letterCount += x;
        waiting = false;
      }, 1000)
    } else if (letterCount === words[0].length + 1 && waiting === false) {
      waiting = true;
      window.setTimeout(function() {
        x = -1;
        letterCount += x;
        waiting = false;
      }, 1000)
    } else if (waiting === false) {
      target.innerHTML = words[0].substring(0, letterCount)
      letterCount += x;
    }
  }, 120)
  window.setInterval(function() {
    if (visible === true) {
      con.className = 'console-underscore hidden'
      visible = false;

    } else {
      con.className = 'console-underscore'

      visible = true;
    }
  }, 400)
}