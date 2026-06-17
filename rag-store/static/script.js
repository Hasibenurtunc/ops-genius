// ── State ────────────────────────────────────────────
let products = [];
let cart = [];
let chatOpen = false;
let cartOpen = false;

// ── Products ─────────────────────────────────────────
async function loadProducts() {
  try {
    const res = await fetch('/api/products');
    products = await res.json();
    renderProducts(products);
  } catch (e) {
    console.error('Failed to load products', e);
  }
}

function renderProducts(list) {
  const grid = document.getElementById('product-grid');
  grid.innerHTML = list.map(p => `
    <div class="product-card" data-category="${p.category}">
      <img class="product-img" src="${p.image}" alt="${p.name}" loading="lazy" />
      <div class="product-info">
        <div class="product-category">${p.category}</div>
        <div class="product-name">${p.name}</div>
        <div class="product-desc">${p.description}</div>
        <div class="product-bottom">
          <span class="product-price">$${p.price.toFixed(2)}</span>
          <button class="add-cart-btn" id="add-btn-${p.id}" onclick="addToCart(${p.id})" aria-label="Add to cart">+</button>
        </div>
      </div>
    </div>
  `).join('');
}

function filterProducts(category, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const filtered = category === 'All' ? products : products.filter(p => p.category === category);
  renderProducts(filtered);
}

// ── Cart ─────────────────────────────────────────────
function addToCart(id) {
  const product = products.find(p => p.id === id);
  if (!product) return;
  const existing = cart.find(c => c.id === id);
  if (existing) { existing.qty++; }
  else { cart.push({ ...product, qty: 1 }); }
  updateCartUI();
  showToast(`✓ ${product.name} added to cart`);
  const btn = document.getElementById(`add-btn-${id}`);
  if (btn) { btn.classList.add('added'); btn.textContent = '✓'; setTimeout(() => { btn.classList.remove('added'); btn.textContent = '+'; }, 800); }
}

function removeFromCart(id) {
  cart = cart.filter(c => c.id !== id);
  updateCartUI();
}

function updateCartUI() {
  document.getElementById('cart-count').textContent = cart.reduce((s, c) => s + c.qty, 0);
  const total = cart.reduce((s, c) => s + c.price * c.qty, 0);
  document.getElementById('cart-total-price').textContent = `$${total.toFixed(2)}`;
  const container = document.getElementById('cart-items');
  if (cart.length === 0) {
    container.innerHTML = '<div class="cart-empty"><span>🛒</span>Your cart is empty</div>';
    return;
  }
  container.innerHTML = cart.map(c => `
    <div class="cart-item">
      <img src="${c.image}" alt="${c.name}" />
      <div class="cart-item-info">
        <h4>${c.name}</h4>
        <p>Qty: ${c.qty}</p>
      </div>
      <span class="cart-item-price">$${(c.price * c.qty).toFixed(2)}</span>
      <button class="cart-item-remove" onclick="removeFromCart(${c.id})">✕</button>
    </div>
  `).join('');
}

function toggleCart() {
  cartOpen = !cartOpen;
  document.getElementById('cart-sidebar').classList.toggle('open', cartOpen);
  document.getElementById('cart-overlay').classList.toggle('open', cartOpen);
}

// ── Chat ─────────────────────────────────────────────
function toggleChat() {
  chatOpen = !chatOpen;
  document.getElementById('chat-window').classList.toggle('open', chatOpen);
  document.getElementById('chat-fab').textContent = chatOpen ? '✕' : '💬';
  if (chatOpen) document.getElementById('chat-input').focus();
}

function openChat() {
  if (!chatOpen) toggleChat();
}

function appendMessage(role, text) {
  const container = document.getElementById('chat-messages');
  const isUser = role === 'user';
  const html = `
    <div class="msg-row ${isUser ? 'user' : ''}">
      ${isUser ? '' : '<div class="msg-avatar bot">🤖</div>'}
      <div class="msg-bubble ${isUser ? 'user' : 'bot'}">${text}</div>
      ${isUser ? '<div class="msg-avatar user">👤</div>' : ''}
    </div>`;
  container.insertAdjacentHTML('beforeend', html);
  container.scrollTop = container.scrollHeight;
}

function appendSource(source) {
  if (!source || source === 'N/A') return;
  const container = document.getElementById('chat-messages');
  container.insertAdjacentHTML('beforeend', `<div class="msg-source">📄 ${source}</div>`);
  container.scrollTop = container.scrollHeight;
}

function showTyping() {
  const container = document.getElementById('chat-messages');
  container.insertAdjacentHTML('beforeend', `
    <div class="msg-row" id="typing-row">
      <div class="msg-avatar bot">🤖</div>
      <div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>
    </div>`);
  container.scrollTop = container.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing-row');
  if (el) el.remove();
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  appendMessage('user', text);
  showTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text })
    });
    const data = await res.json();
    hideTyping();
    appendMessage('bot', data.answer);
    appendSource(data.source);
  } catch (e) {
    hideTyping();
    appendMessage('bot', 'Sorry, something went wrong. Please try again.');
  }
}

function sendQuick(question) {
  document.getElementById('chat-input').value = question;
  sendMessage();
}

// ── Toast ────────────────────────────────────────────
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

// ── Init ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadProducts();
  updateCartUI();
});
