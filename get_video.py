import yt_dlp

url = 'https://www.youtube.com/watch?app=desktop&v=5fU1M1Fxa70'

# ダウンロードするディレクトリを指定
output_path = 'data/video'

# yt-dlpのオプションを設定
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',  # 最高品質の動画と音声を選択
    'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # 出力ファイルのテンプレート
}

# 動画のダウンロード
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])