# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. Support For Thai language.

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/Peng%20Cheng-Lab-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

เครื่องมือเปลี่ยนข้อความเป็นคำพูดภาษาไทย Zero-shot TTS ด้วยโมเดล F5-TTS
โมเดล Finetune : [VIZINTZOR/F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)
 - ชุดข้อมุลในการเทรน : [Porameht/processed-voice-th-169k](https://huggingface.co/datasets/Porameht/processed-voice-th-169k)
 - จำนวน 40,000 เสียง ประมาณ 50 ชั่วโมง
 - โมเดล steps : 150,000
   
# การติดตั้ง
```sh
git clone https://github.com/VYNCX/F5-TTS-THAI.git
cd F5-TTS-THAI
python -m venv venv
call venv/scripts/activate
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

#จำเป็นต้องติดตั้งเพื่อใช้งานได้มีประสิทธิภาพกับ GPU
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

# การใช้งาน
สามารถรันไฟล์ `app-webui.py` เพื่อใช้งานได้ หรือ 

```sh
  python src/f5_tts/f5_tts_webui.py
```

