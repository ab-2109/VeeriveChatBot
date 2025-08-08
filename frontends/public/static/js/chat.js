// Variables to track state
let waitingForClarification = false;
let activeSessionId = null;

// === GLOBAL STATE (add near top if not present) ===
let isWaitingForResponse = false;

const API_BASE = '/api'; // Update with your actual API base URL
const DEBUG = true;

// Debug function
function debug(label, data) {
    if (!DEBUG) return;
    console.log(`DEBUG - ${label}:`, data);
    
    // Also show in UI if we have a debug panel
    const debugPanel = document.getElementById('debug-panel');
    if (debugPanel) {
        const entry = document.createElement('div');
        entry.innerHTML = `<strong>${label}</strong>: ${JSON.stringify(data, null, 2)}`;
        debugPanel.appendChild(entry);
        debugPanel.scrollTop = debugPanel.scrollHeight;
    }
}


// --- Simple Paginator for PDF sections ---
function makePaginator(items, renderItem, opts = {}) {
    const pageSize = opts.pageSize || 4;
    const title = opts.title || '';
    const container = document.createElement('div');
    container.className = 'pdf-paginator';

    const header = document.createElement('div');
    header.className = 'pdf-paginator-header';
    if (title) {
        const h = document.createElement('h4');
        h.textContent = title;
        header.appendChild(h);
    }
    container.appendChild(header);

    const list = document.createElement('div');
    list.className = 'pdf-paginator-list';
    container.appendChild(list);

    const controls = document.createElement('div');
    controls.className = 'pdf-paginator-controls';
    container.appendChild(controls);

    let page = 0;
    const totalPages = Math.max(1, Math.ceil(items.length / pageSize));

    function renderPage() {
        list.innerHTML = '';
        const start = page * pageSize;
        const end = Math.min(items.length, start + pageSize);
        for (let i = start; i < end; i++) {
            const itemEl = renderItem(items[i], i);
            list.appendChild(itemEl);
        }
        controls.innerHTML = '';
        const info = document.createElement('span');
        info.className = 'pdf-page-info';
        info.textContent = `Page ${page + 1} / ${totalPages}`;
        controls.appendChild(info);

        const prev = document.createElement('button');
        prev.textContent = 'Prev';
        prev.disabled = page === 0;
        prev.onclick = () => { page = Math.max(0, page - 1); renderPage(); };
        controls.appendChild(prev);

        const next = document.createElement('button');
        next.textContent = 'Next';
        next.disabled = page >= totalPages - 1;
        next.onclick = () => { page = Math.min(totalPages - 1, page + 1); renderPage(); };
        controls.appendChild(next);
    }

    // If only a few items, don't show controls but still render
    renderPage();
    return container;
}
// Core functions for the chat interface
document.addEventListener('DOMContentLoaded', () => {
    
/* --- Minimal styles for PDF paginator --- */
const styleEl = document.createElement('style');
styleEl.textContent = `
.pdf-paginator { margin: 8px 0 16px; border: 1px solid #333; border-radius: 8px; padding: 8px; }
.pdf-paginator-header h4 { margin: 6px 0 10px; }
.pdf-paginator-list .pdf-text-excerpt, 
.pdf-paginator-list .pdf-table-excerpt { margin-bottom: 8px; }
.pdf-paginator-controls { display: flex; gap: 10px; align-items: center; margin-top: 8px; }
.pdf-paginator-controls button { padding: 4px 8px; cursor: pointer; }
.pdf-page-info { opacity: 0.8; margin-right: auto; }
`;
document.head.appendChild(styleEl);


    // Create debug panel
    const debugPanel = document.createElement('div');
    debugPanel.id = 'debug-panel';
    debugPanel.style = 'position: fixed; bottom: 10px; right: 10px; width: 300px; height: 200px; background: #f5f5f5; border: 1px solid #ccc; overflow: auto; padding: 10px; font-size: 11px; z-index: 9999;';
    document.body.appendChild(debugPanel);
    
    // Get DOM elements
    const messagesContainer = document.getElementById('messages-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    console.log('Send button found:', !!sendButton);
    
    // Test API connection
    fetch(`${API_BASE}/`)
        .then(res => res.json())
        .then(data => {
            debug('API Connection Success', data);
        })
        .catch(err => {
            debug('API Connection Error', err.message);
        });
    
    // Attach event listener to send button
    sendButton.addEventListener('click', (e) => {
        console.log('Button clicked!');
        if (waitingForClarification) {
            submitClarification();
        } else {
            sendQuestion();
        }
    });
    
    // Auto resize the textarea as user types
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = questionInput.scrollHeight + 'px';
    });
    
    // Send message on Enter (but allow Shift+Enter for new lines)
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (waitingForClarification) {
                submitClarification();
            } else {
                sendQuestion();
            }
        }
    });
});

// Send a new question
function sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question) return;

    addMessage(question, 'user');
    input.value = '';
    disableInput();

    debug('Sending question', { query: question });
    
    fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            query: question,
            metadata: {}
        })
    })
    .then(res => {
        debug('Response status', { status: res.status });
        return res.json();
    })
    .then(data => {
        debug('Response data', data);
        processApiResponse(data);
    })
    .catch(err => {
        debug('Error', err);
        addMessage(`Error: ${err.message}`, 'system');
        resetConversationState();
    });
}

// Submit a clarification answer
function submitClarification() {
    const input = document.getElementById('question-input');
    const answer = input.value.trim();
    if (!answer || !activeSessionId) return;

    addMessage(answer, 'user');
    input.value = '';
    input.placeholder = 'Processing...';
    disableInput();
    
    debug('Submitting clarification', { 
        session_id: activeSessionId,
        answer: answer
    });

    fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: answer,
            session_id: activeSessionId
        })
    })
    .then(res => {
        debug('Clarification response status', { status: res.status });
        return res.json();
    })
    .then(data => {
        debug('Clarification response data', data);
        processApiResponse(data);
    })
    .catch(err => {
        debug('Clarification error', err);
        addMessage(`Error: ${err.message}`, 'system');
        resetConversationState();
    });
}

// Process API response with comprehensive checks for all possible formats
function processApiResponse(data) {
    debug('Processing API response', data);
    
    // Check for clarification needed status or questions
    if (data.status === "clarification_needed" || data.clarification_question) {
        handleClarificationNeeded(data);
    } 
    else if (data.status === "complete") {
        handleCompletedResponse(data);
    }
    else if (data.status === "error") {
        const errorMsg = data.error || "Unknown error occurred";
        addMessage(`Error: ${errorMsg}`, 'system');
        resetConversationState();
    }
    else {
        addMessage(`Error: Received unknown response from server`, 'system');
        resetConversationState();
    }
}

// Handle clarification needed
function handleClarificationNeeded(data) {
    activeSessionId = data.session_id;
    waitingForClarification = true;
    
    // Find clarification question - prioritize direct field
    const question = data.clarification_question || 
                     (data.result && data.result.clarification_question) ||
                     "Please provide additional information";
    
    debug('Found clarification question', { question });
    
    // Display the question to the user
    addMessage(question, 'system');
    
    // Update UI to show we're awaiting clarification
    document.getElementById('send-button').textContent = "Submit";
    document.getElementById('question-input').placeholder = "Provide additional information...";
    enableInput();
    document.getElementById('question-input').focus();
}

// Handle completed response
// Handle completed response
function handleCompletedResponse(data) {
    debug("Raw response data", data);
    waitingForClarification = false;
    activeSessionId = null;

    // Reset UI elements
    document.getElementById('send-button').textContent = "Send";
    document.getElementById('question-input').placeholder = "Ask a finance question...";
    
    let responseData = data.result;
    
    // Try to parse if response is a string
    if (typeof responseData === 'string') {
        try {
            responseData = JSON.parse(responseData);
            debug("Successfully parsed response string to object", responseData);
        } catch (e) {
            debug("JSON parse failed, keeping as string", e);
        }
    }
    
    // Debug log all response data types
    debug("Response data type", typeof responseData);
    if (typeof responseData === 'object' && responseData !== null) {
        debug("Response data keys", Object.keys(responseData));
        if (responseData.conversational) {
            debug("Conversational data type", typeof responseData.conversational);
            debug("Conversational data content", responseData.conversational);
        }
        if (responseData.structured) {
            debug("Structured data type", typeof responseData.structured);
            debug("Structured data keys", responseData.structured.data ? Object.keys(responseData.structured.data) : "No data property");
            debug("Structured data content", responseData.structured);
            
            // Detailed debug of structured response
            if (responseData.structured.data) {
                debug("Structured data is an object with data property", responseData.structured.data);
            } else if (typeof responseData.structured === 'string') {
                try {
                    const parsedStructured = JSON.parse(responseData.structured);
                    debug("Parsed structured string to object", parsedStructured);
                } catch (e) {
                    debug("Failed to parse structured string", e.message);
                }
            } else {
                debug("Raw structured data", responseData.structured);
            }
        }
    }
    
    // Create message container
    let messageContainer = document.createElement('div');
    messageContainer.className = 'message assistant combined';
    
    let messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    let hasContent = false;
    
    // Handle different response formats
    if (responseData && typeof responseData === 'object') {
        debug("Processing structured response", responseData);
        
        // --- Process PDF content ---
        if (responseData.pdf || (responseData.pdf_content && responseData.pdf_content.length > 0)) {
            hasContent = true;
            const pdfSection = document.createElement('div');
            pdfSection.className = 'pdf-section';
            if (responseData.pdf) {
                pdfSection.innerHTML = `
                    <h4>📄 Relevant Documents</h4>
                    ${renderPDFObject(responseData.pdf)}
                `;
            } else {
                const pdfContent = responseData.pdf_content;
                pdfSection.innerHTML = `
                    <h4>📄 Relevant Documents</h4>
                    <div class="pdf-content">${markdownToHtml(pdfContent)}</div>`;
            }
            messageContent.appendChild(pdfSection);
        }
        // --- Process conversational content ---
        if (responseData.conversational) {
            hasContent = true;
            let conversationalContent;
            
            if (typeof responseData.conversational === 'string') {
                conversationalContent = responseData.conversational;
            } else if (typeof responseData.conversational === 'object' && responseData.conversational !== null) {
                if (responseData.conversational.data) {
                    conversationalContent = responseData.conversational.data;
                } else {
                    conversationalContent = JSON.stringify(responseData.conversational);
                }
            } else {
                conversationalContent = "No conversational data available";
            }
            
            const conversationalDiv = document.createElement('div');
            conversationalDiv.className = 'conversational-part';
            conversationalDiv.innerHTML = markdownToHtml(conversationalContent);
            messageContent.appendChild(conversationalDiv);
        }
        
        // --- Process structured content ---
        if (responseData.structured && typeof responseData.structured === 'object') {
            hasContent = true;
            let structuredData;
            
            // Extract the actual structured data
            if (responseData.structured.data) {
                structuredData = responseData.structured.data;
            } else {
                structuredData = responseData.structured;
            }
            
            // Skip if structured data is empty
            if (structuredData && Object.keys(structuredData).length > 0) {
                // Add separator if we already have content
                if (messageContent.childNodes.length > 0) {
                    const separator = document.createElement('hr');
                    separator.className = 'content-separator';
                    messageContent.appendChild(separator);
                }
                
                const structuredDiv = document.createElement('div');
                structuredDiv.className = 'structured-response';

                const header = document.createElement('h3');
                header.textContent = 'Structured Analysis';
                structuredDiv.appendChild(header);

                // Render modules as dropdown accordions in order
                const order = ['module1','module2','module3','module4'];
                order.forEach((mod, idx) => {
                    if (structuredData[mod]) {
                        const acc = renderModuleAccordion(mod, structuredData[mod], idx===0);
                        structuredDiv.appendChild(acc);
                    }
                });

                messageContent.appendChild(structuredDiv);
            }
        }
        
        // Fallback for other object formats
        if (!hasContent) {
            debug("No structured or conversational content found, using fallback");
            const fallbackDiv = document.createElement('div');
            fallbackDiv.className = 'fallback-content';
            fallbackDiv.textContent = JSON.stringify(responseData, null, 2);
            messageContent.appendChild(fallbackDiv);
            hasContent = true;
        }
    } 
    // Handle simple string responses
    else if (typeof responseData === 'string') {
        debug("Processing string response");
        const textDiv = document.createElement('div');
        textDiv.className = 'text-content';
        textDiv.innerHTML = markdownToHtml(responseData);
        messageContent.appendChild(textDiv);
        hasContent = true;
    }
    
    // Final fallback if no content was added
    if (!hasContent) {
        debug("No content found, using empty fallback");
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'empty-content';
        emptyDiv.textContent = "No response content available. Please try again.";
        messageContent.appendChild(emptyDiv);
    }
    
    // Add message to container and display
    messageContainer.appendChild(messageContent);
    document.getElementById('messages-container').appendChild(messageContainer);
    
    // Auto-scroll to bottom
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    enableInput();
    document.getElementById('question-input').focus();
}

// --- Helper: Render Table ---
function renderTable(tableData) {
    if (!Array.isArray(tableData) || tableData.length === 0) return '';
    let html = '<table class="response-table"><thead><tr>';
    // Table headers from keys of first row
    Object.keys(tableData[0]).forEach(key => {
        html += `<th>${key}</th>`;
    });
    html += '</tr></thead><tbody>';
    tableData.forEach(row => {
        html += '<tr>';
        Object.values(row).forEach(val => {
            html += `<td>${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

// --- Fix renderPDFObject function ---
function renderPDFObject(pdfObj) {
    if (!pdfObj) return '';
    const { tables = [], texts = [], combined_markdown = '' } = pdfObj;
    const container = document.createElement('div');
    container.className = 'pdf-content';

    // Helper to render an item node from a record
    function renderTextItem(rec, idx) {
        const div = document.createElement('div');
        div.className = 'pdf-text-excerpt';
        const safe = (rec && (rec.content || rec)) || '';
        div.innerHTML = `<div class="pdf-chunk"><strong>Text ${idx + 1}:</strong> ${markdownToHtml(typeof safe === 'string' ? safe : '')}</div>`;
        if (rec && rec.title) {
            const meta = document.createElement('div');
            meta.className = 'pdf-meta';
            meta.textContent = rec.title;
            div.appendChild(meta);
        }
        return div;
    }

    function renderTableItem(rec, idx) {
        const div = document.createElement('div');
        div.className = 'pdf-table-excerpt';
        const safe = (rec && (rec.content || rec)) || '';
        div.innerHTML = `<div class="pdf-chunk"><strong>Table ${idx + 1}:</strong> ${markdownToHtml(typeof safe === 'string' ? safe : '')}</div>`;
        if (rec && rec.title) {
            const meta = document.createElement('div');
            meta.className = 'pdf-meta';
            meta.textContent = rec.title;
            div.appendChild(meta);
        }
        return div;
    }

    if (texts.length) {
        const textPager = makePaginator(texts, renderTextItem, { title: '📄 PDF Text Excerpts', pageSize: 4 });
        container.appendChild(textPager);
    }

    if (tables.length) {
        const tablePager = makePaginator(tables, renderTableItem, { title: '📊 Tables', pageSize: 4 });
        container.appendChild(tablePager);
    }

    if (!texts.length && !tables.length && combined_markdown) {
        const div = document.createElement('div');
        div.innerHTML = markdownToHtml(combined_markdown);
        container.appendChild(div);
    }

    return container.outerHTML;
}

// --- Proper rendering function for PDF sections ---
function renderPDFSection(pdfContent) {
    if (!pdfContent || pdfContent === '') {
        return '';
    }
    
    return `
        <div class="pdf-content">
            <h3>📄 Relevant Documents</h3>
            <div class="pdf-text">${markdownToHtml(pdfContent)}</div>
        </div>
    `;
}

// --- Helper: Render a Module ---


// --- Structured Rendering Helpers ---
function renderTable(obj) {
    try {
        const headers = obj.headers || [];
        const rows = obj.rows || [];
        let html = '<table class="data-table"><thead><tr>';
        headers.forEach(h => { html += `<th>${escapeHtml(String(h))}</th>`; });
        html += '</tr></thead><tbody>';
        rows.forEach(r => {
            html += '<tr>' + r.map(c => `<td>${escapeHtml(String(c))}</td>`).join('') + '</tr>';
        });
        html += '</tbody></table>';
        return html;
    } catch (_) {
        return `<pre>${escapeHtml(JSON.stringify(obj, null, 2))}</pre>`;
    }
}

function tryParseJson(text) {
    if (typeof text !== 'string') return null;
    try {
        return JSON.parse(text);
    } catch {
        // Try to extract ```json ... ```
        const m = text.match(/```json\s*([\s\S]*?)\s*```/i);
        if (m) {
            try { return JSON.parse(m[1]); } catch {}
        }
        return null;
    }
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function renderValue(val) {
    // Strings that look like tables-in-JSON
    if (typeof val === 'string') {
        const maybe = tryParseJson(val);
        if (maybe && (maybe.headers || maybe.rows)) return renderTable(maybe);
        return `<p>${markdownToHtml(val)}</p>`;
    }
    // Arrays
    if (Array.isArray(val)) {
        // array of objects/tables/strings
        let html = '<ul>';
        val.forEach(item => {
            if (item && (item.headers || item.rows)) {
                html += '<li>' + renderTable(item) + '</li>';
            } else if (typeof item === 'object') {
                html += '<li><pre>' + escapeHtml(JSON.stringify(item, null, 2)) + '</pre></li>';
            } else {
                html += '<li>' + markdownToHtml(String(item)) + '</li>';
            }
        });
        html += '</ul>';
        return html;
    }
    // Objects that look like tables
    if (val && (val.headers || val.rows)) return renderTable(val);
    // Generic object
    if (typeof val === 'object' && val !== null) {
        let html = '';
        for (const k in val) {
            const pretty = k.replace(/_/g,' ').replace(/([A-Z])/g,' $1').trim();
            html += `<div class="section"><h5>${escapeHtml(pretty)}</h5>${renderValue(val[k])}</div>`;
        }
        return html;
    }
    // Fallback
    return `<p>${escapeHtml(String(val ?? ''))}</p>`;
}

function renderModuleAccordion(moduleName, moduleData, open=false) {
    const title = moduleName.replace(/([A-Z])/g,' $1').trim();
    const det = document.createElement('details');
    det.className = 'dropdown-section';
    if (open) det.setAttribute('open','open');

    const sum = document.createElement('summary');
    sum.innerHTML = `<h4>${escapeHtml(title)}</h4>`;
    det.appendChild(sum);

    const inner = document.createElement('div');
    inner.className = 'structured-content';
    // If the module is string, show as markdown; if object, render sections
    if (typeof moduleData === 'string') {
        inner.innerHTML = markdownToHtml(moduleData);
    } else if (typeof moduleData === 'object' && moduleData !== null) {
        for (const key in moduleData) {
            const label = key.replace(/_/g,' ').replace(/([A-Z])/g,' $1').trim();
            const section = document.createElement('div');
            section.className = 'section';
            section.innerHTML = `<h4>${escapeHtml(label)}</h4>${renderValue(moduleData[key])}`;
            inner.appendChild(section);
        }
    } else {
        inner.innerHTML = `<p>${escapeHtml(String(moduleData ?? ''))}</p>`;
    }
    det.appendChild(inner);
    return det;
}

function renderModule(moduleName, moduleData) {
    const moduleElement = document.createElement('div');
    moduleElement.className = 'module';
    
    // Create readable module title
    const title = moduleName
        .replace(/([A-Z])/g, ' $1') // Add space before capital letters
        .replace(/module\d+/i, '') // Remove "module1", "module2", etc.
        .trim();
    
    let moduleContent = `<h3>${title || moduleName}</h3>`;
    
    // Process module content based on data type
    for (const key in moduleData) {
        if (moduleData.hasOwnProperty(key)) {
            const value = moduleData[key];
            
            // Handle special case for tables
            if (key === 'model_comparison' || (typeof value === 'object' && value.headers && value.rows)) {
                moduleContent += renderTable(value);
            }
            // Handle arrays
            else if (Array.isArray(value)) {
                moduleContent += `<h4>${formatKey(key)}</h4><ul>`;
                value.forEach(item => {
                    moduleContent += `<li>${item}</li>`;
                });
                moduleContent += `</ul>`;
            }
            // Handle text content
            else if (typeof value === 'string') {
                moduleContent += `<h4>${formatKey(key)}</h4>
                                 <p>${markdownToHtml(value)}</p>`;
            }
        }
    }
    
    moduleElement.innerHTML = moduleContent;
    return moduleElement;
}

// Format object keys to be more readable
function formatKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, letter => letter.toUpperCase());
}

// Add this helper function for markdown conversion
function markdownToHtml(markdown) {
    if (!markdown) return "";
    
    // Replace headers
    let html = markdown
        .replace(/##### (.*?)(\n|$)/g, '<h5>$1</h5>')
        .replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>')
        .replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>')
        .replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>')
        .replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>');
    
    // Replace lists
    html = html.replace(/^\s*[*-] (.*?)(\n|$)/gm, '<li>$1</li>');
    html = html.replace(/<li>(.*?)<\/li>(\n<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    // Replace paragraphs
    html = html.replace(/^(?!<[uh]|<li)(.*?)(\n|$)/gm, '<p>$1</p>');
    
    return html;
}

// Add additional debug helper
function debugPrintObject(obj, label) {
    try {
        debugDisplay(`${label}: ${JSON.stringify(obj, null, 2).substring(0, 500)}...`);
    } catch (e) {
        debugDisplay(`${label}: [Error stringifying object: ${e.message}]`);
    }
}

// Update the addMessage function to support the "structured" class
function addMessage(content, sender) {
    const messagesContainer = document.getElementById('messages-container');
    
    const messageDiv = document.createElement('div');
    
    // Check if sender has additional classes (like "assistant structured")
    const senderClasses = sender.split(' ');
    const primarySender = senderClasses[0]; // "assistant" in "assistant structured"
    
    // Set the base class
    messageDiv.className = `message ${primarySender}`;
    
    // Add any additional classes
    if (senderClasses.length > 1) {
        for (let i = 1; i < senderClasses.length; i++) {
            messageDiv.classList.add(senderClasses[i]);
        }
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (typeof content === 'string') {
        messageContent.innerHTML = content;
    } else {
        messageContent.innerHTML = JSON.stringify(content, null, 2);
    }
    
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// === INPUT CONTROL HELPERS (ADD BEFORE sendQuestion / submitClarification) ===
function getInputEl() {
    return document.getElementById('question-input');
}
function getSendButton() {
    return document.getElementById('send-button');
}

function disableInput() {
    const input = getInputEl();
    const btn = getSendButton();
    if (input) {
        input.disabled = true;
        input.classList.add('disabled');
    }
    if (btn) {
        btn.disabled = true;
        btn.classList.add('disabled');
    }
    isWaitingForResponse = true;
}

function enableInput() {
    const input = getInputEl();
    const btn = getSendButton();
    if (input) {
        input.disabled = false;
        input.classList.remove('disabled');
    }
    if (btn) {
        btn.disabled = false;
        btn.classList.remove('disabled');
    }
    isWaitingForResponse = false;
}

// Optional visual busy indicator
function setThinking(on) {
    const btn = getSendButton();
    if (btn) {
        btn.textContent = on ? '...' : (waitingForClarification ? 'Submit' : 'Send');
    }
}

// === FIX resetConversationState (replace existing) ===
function resetConversationState() {
    activeSessionId = '';
    waitingForClarification = false;
    isWaitingForResponse = false;

    // Use correct container id
    const messagesContainer = document.getElementById('messages-container');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }

    // Use correct input id
    const inputField = document.getElementById('question-input');
    if (inputField) {
        inputField.value = '';
        inputField.placeholder = 'Ask a finance question...';
    }

    // Restore send button
    const sendBtn = document.getElementById('send-button');
    if (sendBtn) {
        sendBtn.textContent = 'Send';
        sendBtn.disabled = false;
    }

    enableInput();
    addMessage('How can I help you today?', 'assistant');
    console.log('Conversation state reset');
}

// === (Optional) Attach reset button safely (keep existing or replace) ===
document.addEventListener('DOMContentLoaded', () => {
    const resetButton = document.getElementById('reset-button');
    if (resetButton && !resetButton.dataset.bound) {
        resetButton.dataset.bound = '1';
        resetButton.addEventListener('click', resetConversationState);
    }
});