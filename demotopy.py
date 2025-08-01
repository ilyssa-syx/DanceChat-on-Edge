# Output Directories
output_folder = "custom_music"
motion_folder = "SMPL-to-FBX/motions"
fbx_folder = "SMPL-to-FBX/fbx_out"
urllist = [
    "https://www.youtube.com/watch?v=2RicaUqd9Hg",
    "https://www.youtube.com/watch?v=-CCgDvUM4TM",
    "https://www.youtube.com/watch?v=9KhbM2mqhCQ",
    "https://www.youtube.com/watch?v=s3bksUSPB4c",
    "https://www.youtube.com/watch?v=ABfQuZqq8wg",
    "https://www.youtube.com/watch?v=fNFzfwLM72c",
    "https://www.youtube.com/watch?v=9i6bCWIdhBw",
    "https://www.youtube.com/watch?v=yURRmWtbTbo",
    "https://www.youtube.com/watch?v=god7hAPv8f0",
    "https://www.youtube.com/watch?v=1sqE6P3XyiQ",
    "https://www.youtube.com/watch?v=Zi_XLOBDo_Y",
    "https://www.youtube.com/watch?v=UHXGc2oWyJ4",
    "https://www.youtube.com/watch?v=h4bP9tj_0Zk",
    "https://www.youtube.com/watch?v=qK5KhQG06xU",
    "https://www.youtube.com/watch?v=FDMHZFJnk2s",
    "https://www.youtube.com/watch?v=4NJH75q0Syk",
    "https://www.youtube.com/watch?v=ggJI9dKBk48",
    "https://www.youtube.com/watch?v=LPYw3jXjd74",
    "https://www.youtube.com/watch?v=P-sGt5E2epc",
    "https://www.youtube.com/watch?v=GxBSyx85Kp8",
    "https://www.youtube.com/watch?v=OPf0YbXqDm0",
    "https://www.youtube.com/watch?v=c18441Eh_WE",
    "https://www.youtube.com/watch?v=uSD4vsh1zDA",
    "https://www.youtube.com/watch?v=ViwtNLUqkMY",
    "https://www.youtube.com/watch?v=q0KZuZF01FA",
    "https://www.youtube.com/watch?v=Vds8ddYXYZY",
    "https://www.youtube.com/watch?v=gm3-m2CFVWM",
    "https://www.youtube.com/watch?v=BerNfXSuvJ0",
    "https://www.youtube.com/watch?v=aFmTvY11vug",
    "https://www.youtube.com/watch?v=LOZuxwVk7TU",
    "https://www.youtube.com/watch?v=TUVcZfQe-Kw",
    "https://www.youtube.com/watch?v=Ab6E2BsuLJ0",
    "https://www.youtube.com/watch?v=g7X9X6TlrUo",
    "https://www.youtube.com/watch?v=nsXwi67WgOo",
    "https://www.youtube.com/watch?v=HCq1OcAEAm0",
]
import os
os.makedirs(output_folder, exist_ok=True)

for url in urllist:
    os.system(f'yt-dlp --extract-audio --audio-format wav --audio-quality 0 --output "{output_folder}/%(id)s.%(ext)s" "{url}"')