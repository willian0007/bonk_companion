# bonk companion

**vibe coding project remake Ai avatar companion of grok to run on local**

99% โปรเจกต์ไวป์โคดดิ้งสร้าง เอไออวาตาร์แบบกร๊วก เอ้ย กร๊อก ให้รันบนเครื่องตัวเองได้ เอไออวาตาร์เลียนแบบ grok companion รันผ่าน local โดยใช้ ue5.6+whisper+gemini api+f5tts+neurosync

grok companion ของ elon musk รันผ่าน server และต้องจ่าย 1000 บาทเพื่อขอให้มิสะ เอ้ย อนิ เปลื้องผ้า โปรเจกต์ของผมที่เริ่มทำมา 2 เดือน สามารถปรับ customize ได้ทุกอย่าง (ถ้าทำสำเร็จ Lol) 

หลักการคือ ใช้ STT เอาคำพูดแปลงเป็นข้อความ ส่งไปยัง gemini api ที่ตอนนี้เรทฟรีเยอะ (20/7/68) แล้วรัน response text ผ่าน TTS F5TTS thai โมเดล opensource ของคนไทยที่เทรนด้วยข้อมูลภาษาไทย 1000 ชม.+- ทำให้สามารถเปลี่ยนเสียงของเอไอได้ด้วย สุดท้ายไปรันเข้าโมเดล lipsync ที่เทรนจากการทำ face livelink ผ่านกล้องไอโฟนเทียบกับ blendshape csv file

คำเตือน*****
1. โปรเจกต์ไม่สมบูรณ์ที 100% มันสู้มิสะของ grok ไม่ได้อยู่แล้ว แต่มันสร้าง path ไปได้หลายทาง lol
2. nvidia supremacy โปรเจกต์นี้ รันผ่าน cuda ทำให้คนที่ใช้ amd น่าจะไม่สามารถใช้ได้ (อาจจะใช้ได้ก็ได้ ถ้าคุณเก่งจริง ลองถาม gemini ดู lol)
3. กินสเป็ก ใช้ vram เยอะ แต่ก็พอรันได้ถ้าปรับดีๆ ผมมีเครื่อง rtx3060 6gb vram laptop อยู่ตัวนึง สมัย 4-5 ปีที่แล้วได้ ยังรันโปรเจกต์นี้ได้
4. ใช้เวลา set up โปรเจกต์ประมาณ 1-2 ชม. น่าจะได้ แต่ถ้าคอมดี เน็ตดี รวมๆ 1ชม. เหลือๆ
## Performance

ทั้งหมดนี้ response time (เวลารอเอไอขยับปากตอบกลับมา) บน rtx5090 ใช้ไปประมาณ 5-8 วิ ต่อ chunk size (รอ gemini โมเดล lite ที่ตอบไวสุดประมาณ 2 วิ whisper 3 large ประมาณ 1 วิ f5TTS batch นึงใช้เวลาประมาณ 2-3 วิ และ blenshape model 2 วิโดยประมาณ )

เทสรันผ่าน rtx3060 6gb vram laptop ผ่าน ถ้าใช้ whisper base สามารถรันได้ รอประมาณ 12-15 วิ โดยประมาณ

## License

opensource 99% license ของ ue5.6 กับ neurosync ขอรายได้หากทำรายได้มากกว่า 30 ล้านบาท

## Features

ปัจจุบัน สามารถสร้างเอไออวาตาร์ตามใจชอบได้ในเกมด้วย metahuman creator ✅ สามารถปรับ ความเร็วเสียง, เสียงอ้างอิง(ref voice) ,system prompt ,STT method whisper หรือ gemini,เลือก โมเดล gemini ได้ในหน้าเมนูออปชั่นเกม ✅

เปลี่ยนแมพ เปลี่ยน asset เปลี่ยนกราฟฟิก เปลี่ยนออปชั่น setting ทำได้หมด ✅

ดัดแปลงอื่นๆ ทั้งโค้ด ทั้งในตัวเกมได้หมด ✅

## สิ่งที่ต้องพัฒนาต่อ

ยังไม่ได้สร้างเป็น .exe file ที่เป็นแอปเกมที lol ก็คือยังเป็น .uproject ที่ต้องรันผ่าน ue5 editor เพราะผมว่า ผมควรปล่อยตัวที่ให้คนเอาไปพัฒนาต่อได้ (พูดเอาหล่อ) กับ อีกเหตุผลคือ ผมไม่ได้ set ให้ตัวเกมเต็มสามารถตั้งค่า livelink ตอนเป็น standalone ได้ที (ปัญหาจริงๆ lol) ซึ่งผมค่อนข้างมั่นใจว่ามันทำได้ แต่แค่ต้องไป setting นิดๆหน่อยๆ 🤔

ตั้งค่ากราฟฟิกในเกม และ อื่นๆ ที่จะช่วยลดความหน่วงได้ 🤔

ปรับ animation sequence ปรับ interactive อื่นๆ ปรับ ui main menu blah blah ยังไม่ได้ลอง ยังไม่ได้ฝึก lol แต่คิดว่าน่าจะทำได้อยู่ ใครทำได้ทำเลย 🤔

subtitle ยังทำไม่ได้ แต่ไม่น่าจะยากเกินแก้ Lol ใครแก้ได้แก้เลย 🤔

chunk streaming f5TTS ยังไม่เร็วพอ ถ้าจะทำให้ ai ตอบไวๆเลย ถ้าเทรนโมเดลที่สามารถ streaming ได้แบบ kokoro + run local llm จะไวกว่านี้และรันบนเครื่องกากๆได้มีประสิทธิภาพมากขึ้น 🤔

finetune f5tts for more........good voice? ไม่แน่ใจว่าสามารถ finetune ให้พูดชัด พูดถูกต้องมากกว่านี้ พูด......กระซิบ ให้มากกว่านี้ได้มากแค่ไหน แต่ F5TTS อาจจะช้าเกินไป และ ขาดอารมณ์ จากประสปการณ์ส่วนตัวที่เคย train vits tts ด้วยการ scrape จาก youtube มา ผมยังไม่ชัวร์ว่าเราทำอะไรได้มากน้อยแค่ไหน กับ dataset ที่มีให้ใช้ในปัจจุบัน + ทุน lol ดังนั้น ที่พอจะเป็นไปได้ น่าจะรอ finetune thai tts opensource สักอันที่อีกหน่อยเขาคงปล่อยกันออกมาอีก 🤔

local llm จริงๆ สามารถใช้ local llm ได้ แต่ผมรู้สึกว่า ผลลัพธ์คงไม่ต่างจาก gemini api ที่ตอนนี้ฟรี แถมตอบได้ดีอยู่แล้ว ใครอยากปรับ ปรับได้เลย 🤔

ยังไม่มี memory เนื่องจาก run ผ่าน api เลยไม่มีการจัดเก็บประวัติการพูดทีมั้ง (vibe coding in a nutshell here lol) ถาม gemini มาแล้ว มันบอกทำได้แต่ผมยังไม่ลองที เดี๋ยวจะลองดู 🤔

## วิธีติดตั้ง

### สิ่งที่ต้องมีก่อน
- **git** https://git-scm.com/downloads
- **python version 3.10-3.12** (3.13 ไม่ได้ลอง แต่คาดว่าน่าจะมีปัญหา) https://www.python.org/downloads/
- **cuda12.8** (อันอื่นอาจจะได้ ไม่ได้ลองเทสผ่านบน rtx3060)

### ขั้นตอนการติดตั้ง

มีพร้อมแล้ว สร้างโฟลเดอร์ใหม่มา 1 โฟลเดอร์ สมมติชื่อ bonk
กด cmd+ enter ที่ช่องค้นหา 

```bash
python -m venv venv
```
(อย่าลืม set sys env python path เป็น python version ที่ต้องการ)

ภายใน bonk กด cmd+ enter ที่ช่องค้นหา อีกครั้ง ไม่ก็พิมพ์ต่อจากหน้าเมื่อกี้ได้เลย

```bash
git clone https://github.com/willian0007/bonk_companion.git
```

จากนั้น activate venv 
```bash
venv\scripts\activate
```

```bash
cd bonk_companion
pip install -r requirements.txt
```
จะเจอ error บ้าง ไม่ก็ไม่เจอเลย ไม่ว่ากัน ใช้อีกอันนึงต่อ
```bash
pip install -r requirements2.txt
```
ถ้ามันไม่ได้ install torch ให้ ให้ run 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### ไฟล์ที่ต้องดาวโหลดเพิ่มเติม

#### blendshape model
https://huggingface.co/AnimaVR/NEUROSYNC/tree/main โหลด model.pth เอาไปวางใน **bonk_companion**/utils/model

#### vocos
https://huggingface.co/charactr/vocos-mel-24khz/tree/main
โหลด 2 ไฟล์
- config.yaml
- pytorch_model.bin 

วางใน **bonk_companion**/vocoder

#### whisper model ไหนก็ได้
- whisper largev3/whisper large v3 turbo/whisper base
- https://huggingface.co/openai/whisper-base/tree/main
- https://huggingface.co/openai/whisper-large-v3-turbo/tree/main
- https://huggingface.co/openai/whisper-base/tree/main

ดาวโหลดไฟล์พวกนี้ แล้วเอามาลงในโฟลเดอร์ **bonk_companion**/whisper

**Model weights**
- pytorch_model.bin — contains the actual model weights.

**Model config**
- config.json — defines the model architecture (layers, heads, etc).

**Feature extractor and tokenizer**
- preprocessor_config.json — configuration for feature extraction.
- tokenizer.json — defines the tokenizer vocabulary and rules.
- tokenizer_config.json — tokenizer settings.
- vocab.json — vocabulary used by the tokenizer.
- merges.txt — merge rules for the BPE tokenizer.

**Optional but recommended:**
- generation_config.json — settings for controlling decoding during inference (e.g., beam size, temperature).

#### f5tts thai 
https://huggingface.co/VIZINTZOR/F5-TTS-THAI/tree/main
โหลดไฟล์ .pt อันไหนก็ได้ (fp16 จะโหลดไวกว่าแต่ quality น้อยกว่านิด) เอาไปไว้ใน **bonk_companion**/ckpts

#### โหลด myproject ทั้งยวงจาก google drive ผม Lol
เนื่องจากผมไม่รู้วิธีแชร์ ตัว project ue5 ผมก็เลยขอมักง่ายแบบนี้ไปก่อน lol โหลดไปวางไว้ที่ไหนก็ได้ แล้วก็ install epiclancher+ue5.6 ให้เสร็จ เปิด ue5.6 แล้ว browse เข้าตัวไฟล์ .uproject ได้เลย lol ใครอัพเป็นอัพได้อัพเลย
https://drive.google.com/drive/folders/1_w_kKGqybe7Dr1IDXtyJ7vo6Lugc87BY?usp=sharing
### Configuration

กด edit runserver2.bat
```batch
set "ROBOT_MODEL_PATH=%PROJECT_DIR%ckpts\model_650000_FP16.pt"
set "ROBOT_VOCAB_PATH=%PROJECT_DIR%vocab\vocab.txt"
set "ROBOT_REF_AUDIO_PATH=%PROJECT_DIR%soundtest\welp.wav"
set "GEMINI_API_KEY=ใส่ gemini api key ลงตรงนี้"
set "VOCODER_MODEL_PATH=%PROJECT_DIR%vocoder"
```
แก้ โมเดลที่จะใช้ ref voice และ ใส่ gemini api ลงในนี้
แก้ nfe step อื่นๆ ใน robotmodule.py

## Usage

ดับเบิ้ลคลิก runserver2.bat และ เปิดเกมพร้อมกัน (รันโปรเจกต์นี้ต้องรันพร้อมกับ runserver2.bat)

**all done!!!**

### Controls
- **test blendshape model ในเกม** กดปุ่ม k
- **test talk to ai** กดปุ่ม t
- **test option setting** กดปุ่ม escape แต่ต้องไปตั้งค่าปุ่ม stop pie ใน project ก่อน

## About

สุดท้ายนี้ โปรเจกต์นี้ผมไม่ได้เห็น grok companion ของ elon แล้ว ทำตามเสร็จภายใน 5 วันนะ lol ผมทำมาก่อนการมาของ grok companion ประมาณ 2 เดือน (ถ้านับรวมว่าผมไปติด expedition 33 อยู่เกือบเดือนด้วยก็น่าจะแค่เดือนเดียวเองมั้ง lol) และเคยอยากลองทำมาตั้งแต่ปีที่แล้ว ดังนั้นผมพูดได้เต็มปากเต็มคำว่า "เฮ้ย ไม่ใช่ผมที่คิดทำแบบ elon นะเว้ย ขนาด elon มันยังคิดเหมือนผมเลยเห็นไหม" 

เพียงแต่ว่า......จริงๆที่ผมเลือก unreal engine เพราะผมอยากลองมานานแล้ว + กระแสเกม expedition 33 ทำให้ผมเริ่มหัดทำเกมบ้าง และจริงๆเป้าหมาย ตัวเกมนี้คืออยากทำแนว interactive ที่เป็นภาษาไทย ไม่ก็เอาไปรันบน kiosk สักเครื่องได้ เป็นผู้ช่วย assistant อะไรประมาณนั้น lol 

แต่เมื่อ grok companion ของ elon ออกมา มันจุดชนวนไฟใหม่ให้ผม ผมจะลองไปแนวเดียวกับ elon ดีกว่า ตอนนี้กำลังคิดอยู่ว่า ต้องใช้ tool อะไรบ้าง ส่วนโปรเจกต์นี้ ถ้ามีเวลาว่างๆเพิ่มเติม ผมจะมาทำ + อัพเดท เพิ่มเติมต่อ ใครมีปัญหาการใช้งานอะไรยังไง ทักมาหาผมได้ ที่เฟซ william pikeman

โปรเจกต์นี้ vibe coding 99% โค้ดหลายๆส่วนอาจจะมั่ว blueprint พันกันมึนไปหมด ก็ขออภัยด้วย นี่เป็นอีกสาเหตุนึงที่ผมปล่อยทั้งตัว project ให้คนเอาไปพัฒนาต่อ ผมหวังว่ามันจะจุดประกายให้สักคน 2 คน ไม่มากก็น้อย ได้มีไฟเพิ่มขึ้นมา เหมือนที่ผมมีไฟเพิ่มขึ้นจากการนั่งทำโปรเจกต์พวกนี้ผ่าน vibe coding

## Credits

**ขอบคุณ**
- **VIZINTZOR** กับ โมเดล thai F5TTS ที่สามารถนำไป run local ได้
- **open ai** กับ whisper model
- **gemini api** ที่แจก api ฟรีแบบที่เจ้าอื่นไม่กล้าทำ 
- **epic game** สำหรับ ue5.6 กับ asset อื่นๆ ที่ให้เอามาใช้ฟรีได้ถ้ารายได้ไม่ถึง 30 ล้าน
- **AnimaVR** ตัว lipsync model ตัวนี้ ที่ run free on local ผมเจอแค่ของคนนี้จริงๆ กราบเลย เพราะผมคงไปทำ viseme กันไม่ไหวแน่ๆ



