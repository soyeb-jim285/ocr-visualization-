#!/bin/bash
# =============================================================================
# RunPod GPU Pod Setup Script
# =============================================================================
# Sets up a full dev environment: tmux, vim, fonts, Python deps, and starts
# training in a tmux session.
#
# Usage:
#   curl -sSL <raw-github-url> | bash
#   # or
#   bash runpod-setup.sh
# =============================================================================

set -e

echo "============================================"
echo "  RunPod Environment Setup"
echo "============================================"

# ---------------------------------------------------
# 1. System packages
# ---------------------------------------------------
echo ""
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    tmux \
    vim \
    htop \
    nvtop \
    curl \
    wget \
    unzip \
    git \
    fontconfig \
    locales \
    tree \
    > /dev/null 2>&1

# Set UTF-8 locale
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen 2>/dev/null || true
locale-gen en_US.UTF-8 > /dev/null 2>&1 || true
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "  Done."

# ---------------------------------------------------
# 2. Install Nerd Font (for icons in tmux/vim)
# ---------------------------------------------------
echo ""
echo "[2/7] Installing JetBrains Mono Nerd Font..."
FONT_DIR="/usr/local/share/fonts/NerdFonts"
mkdir -p "$FONT_DIR"
if [ ! -f "$FONT_DIR/JetBrainsMonoNerdFont-Regular.ttf" ]; then
    cd /tmp
    wget -q "https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.tar.xz"
    tar -xf JetBrainsMono.tar.xz -C "$FONT_DIR"
    fc-cache -f "$FONT_DIR"
    rm -f JetBrainsMono.tar.xz
    echo "  Font installed."
else
    echo "  Font already installed."
fi

# ---------------------------------------------------
# 3. Tmux config
# ---------------------------------------------------
echo ""
echo "[3/7] Configuring tmux..."
cat > ~/.tmux.conf << 'TMUX_EOF'
# ------- General -------
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"
set -g history-limit 50000
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
set -s escape-time 0
set -g focus-events on
set -g set-clipboard on

# ------- Mouse -------
set -g mouse on

# ------- Prefix -------
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# ------- Splits -------
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# ------- Pane navigation (vim-style) -------
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ------- Resize panes -------
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# ------- Copy mode (vi) -------
setw -g mode-keys vi
bind -T copy-mode-vi v send-keys -X begin-selection
bind -T copy-mode-vi y send-keys -X copy-selection-and-cancel

# ------- Reload config -------
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# ------- Status bar -------
set -g status-position top
set -g status-interval 5
set -g status-style "bg=#1e1e2e,fg=#cdd6f4"

set -g status-left-length 40
set -g status-left "#[fg=#1e1e2e,bg=#89b4fa,bold]  #S #[fg=#89b4fa,bg=#1e1e2e] "

set -g status-right-length 80
set -g status-right "#[fg=#6c7086] %H:%M #[fg=#89b4fa]│ #[fg=#a6e3a1]GPU: #(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)%% #[fg=#89b4fa]│ #[fg=#f9e2af]Mem: #(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"

# Window tabs
setw -g window-status-format "#[fg=#6c7086] #I:#W "
setw -g window-status-current-format "#[fg=#1e1e2e,bg=#89b4fa,bold] #I:#W "

# Pane borders
set -g pane-border-style "fg=#313244"
set -g pane-active-border-style "fg=#89b4fa"
TMUX_EOF
echo "  Done."

# ---------------------------------------------------
# 4. Vim config
# ---------------------------------------------------
echo ""
echo "[4/7] Configuring vim..."
cat > ~/.vimrc << 'VIM_EOF'
" ------- General -------
set nocompatible
syntax on
filetype plugin indent on
set encoding=utf-8
set fileencoding=utf-8
set termguicolors
set background=dark

" ------- UI -------
set number
set relativenumber
set cursorline
set showmatch
set showcmd
set wildmenu
set wildmode=longest:full,full
set laststatus=2
set signcolumn=yes
set scrolloff=8
set sidescrolloff=8
set colorcolumn=100
set title

" ------- Mouse -------
set mouse=a
set ttymouse=sgr

" ------- Indentation -------
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set smartindent
set autoindent

" ------- Search -------
set hlsearch
set incsearch
set ignorecase
set smartcase

" ------- Performance -------
set updatetime=300
set timeoutlen=500
set lazyredraw
set ttyfast

" ------- Files -------
set noswapfile
set nobackup
set undofile
set undodir=~/.vim/undodir
set autoread

" Create undo dir
silent !mkdir -p ~/.vim/undodir

" ------- Splits -------
set splitbelow
set splitright

" ------- Key mappings -------
let mapleader = " "

" Quick save/quit
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>

" Split navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Clear search highlight
nnoremap <leader>/ :nohlsearch<CR>

" Move lines up/down in visual mode
vnoremap J :m '>+1<CR>gv=gv
vnoremap K :m '<-2<CR>gv=gv

" Keep cursor centered
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz
nnoremap n nzzzv
nnoremap N Nzzzv

" Yank to system clipboard
nnoremap <leader>y "+y
vnoremap <leader>y "+y

" ------- Status line -------
set statusline=
set statusline+=\ %f        " file path
set statusline+=\ %m        " modified flag
set statusline+=\ %r        " readonly flag
set statusline+=%=           " right align
set statusline+=\ %Y        " file type
set statusline+=\ \|\ %l:%c " line:column
set statusline+=\ \|\ %p%%  " percentage

" ------- Colorscheme (built-in dark) -------
colorscheme slate
hi Normal guibg=#1e1e2e ctermbg=235
hi CursorLine guibg=#313244 ctermbg=236
hi StatusLine guifg=#89b4fa guibg=#1e1e2e ctermfg=75 ctermbg=235
hi LineNr guifg=#6c7086 ctermfg=242
hi CursorLineNr guifg=#89b4fa ctermfg=75
hi ColorColumn guibg=#313244 ctermbg=236

" ------- Python specific -------
autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
autocmd FileType python setlocal colorcolumn=100
VIM_EOF
echo "  Done."

# ---------------------------------------------------
# 5. Shell config (bashrc additions)
# ---------------------------------------------------
echo ""
echo "[5/7] Configuring shell..."
cat >> ~/.bashrc << 'BASH_EOF'

# ------- RunPod custom config -------
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export TERM=xterm-256color

# Aliases
alias ll='ls -lah --color=auto'
alias la='ls -A --color=auto'
alias ..='cd ..'
alias ...='cd ../..'
alias gs='git status'
alias gd='git diff'
alias gl='git log --oneline -20'
alias gpu='nvidia-smi'
alias gpuw='watch -n 1 nvidia-smi'
alias ta='tmux attach -t'
alias tl='tmux list-sessions'
alias tn='tmux new -s'
alias py='python3'

# GPU info
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
alias gputemp='nvidia-smi --query-gpu=temperature.gpu --format=csv'

# Quick training monitor
alias trainlog='tail -f /workspace/train.log'

# Better prompt
PS1='\[\e[38;5;75m\][\[\e[38;5;114m\]\u\[\e[38;5;75m\]@\[\e[38;5;212m\]runpod\[\e[38;5;75m\]] \[\e[38;5;229m\]\w\[\e[0m\]\n\$ '
BASH_EOF
echo "  Done."

# ---------------------------------------------------
# 6. Python dependencies
# ---------------------------------------------------
echo ""
echo "[6/7] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    torch \
    torchvision \
    onnxscript \
    numpy \
    tqdm
echo "  Done."

# ---------------------------------------------------
# 7. Verify GPU setup
# ---------------------------------------------------
echo ""
echo "[7/7] Verifying GPU setup..."
python3 << 'PYEOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    torch.set_float32_matmul_precision("high")
    print("  TF32 matmul: enabled")
else:
    print("  WARNING: No GPU detected!")
PYEOF

# ---------------------------------------------------
# Done
# ---------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Quick start:"
echo "    1. Upload train_emnist.py to /workspace/"
echo "    2. tmux new -s train"
echo "    3. python3 train_emnist.py"
echo ""
echo "  Tmux cheatsheet:"
echo "    Prefix:       Ctrl+A"
echo "    Split horiz:  Ctrl+A |"
echo "    Split vert:   Ctrl+A -"
echo "    Navigate:     Ctrl+A h/j/k/l"
echo "    Detach:       Ctrl+A d"
echo "    Reattach:     tmux attach -t train"
echo "    GPU monitor:  nvtop"
echo ""
