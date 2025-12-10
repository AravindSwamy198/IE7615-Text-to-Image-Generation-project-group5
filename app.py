from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json
import gc

app = Flask(__name__)


MODEL_PATH = "model_to_download (2)/best_evaluated_model"


config_path = os.path.join(MODEL_PATH, "best_config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        BEST_CFG = config['best_cfg']  
        NUM_STEPS = config['num_steps'] 
        print(f"✓ Loaded config from best_config.json")
else:
    
    BEST_CFG = 10.0
    NUM_STEPS = 40
    print("⚠ Using default best settings")


if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
    print("🎉 GPU detected! Using CUDA for faster generation")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print("💻 No GPU detected. Using CPU (generation will be slower)")

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


pipe = None

print("=" * 60)
print("🎓 IE7615 18014 Neural Networks/Deep Learning - Group 5")
print("🌟 Stable Diffusion Image Generator")
print("=" * 60)

def load_model():
    """Load the fine-tuned Stable Diffusion model"""
    global pipe
    
    print(f"\n📂 Loading model from: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"⚙️  Settings: CFG={BEST_CFG}, Steps={NUM_STEPS}")
    
    try:
        
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
        
        
        if DEVICE == "cuda":
            pipe = pipe.to(DEVICE)
            pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, 'enable_xformers_memory_efficient_attention') else None
        
   
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
       
        pipe.enable_attention_slicing()
        
        print("✅ Model loaded successfully!")
        print(f"📊 Using optimized settings from evaluation:")
        print(f"   - FID Score: 153.80")
        print(f"   - Inception Score: 1.0046")
        print(f"   - Scheduler: Euler")
        print(f"   - CFG: {BEST_CFG}, Steps: {NUM_STEPS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image with exact Jupyter notebook settings"""
    global pipe
    
    if pipe is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please refresh the page.'
        }), 500
    
    try:
        # Get prompt from request
        data = request.json
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Please provide a prompt.'
            }), 400
        
        print(f"\n🎨 Generating image: {prompt}")
        print(f"   Resolution: 512x512")
        print(f"   Steps: {NUM_STEPS}")
        print(f"   CFG Scale: {BEST_CFG}")
        print(f"   Scheduler: Euler")
        
        # Clear memory before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate the image - EXACT SAME AS JUPYTER
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,  
                guidance_scale=BEST_CFG,       
                height=512,                      
                width=512,                     
                
            )
        
        image = result.images[0]
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        
        image.save(filepath, "PNG", optimize=False)
        print(f"✅ Saved: {filename} (512x512)")
        
       
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
       
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generation_time = datetime.now() - datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        print(f"⏱️  Generation time: {generation_time.total_seconds():.1f} seconds")
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename,
            'prompt': prompt
        })
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({
            'success': False,
            'error': 'GPU out of memory. Try a simpler prompt or restart the application.'
        }), 500
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    info = {
        'model_loaded': pipe is not None,
        'device': DEVICE,
        'cfg_scale': BEST_CFG,
        'num_steps': NUM_STEPS,
        'scheduler': 'Euler',
        'resolution': '512x512'
    }
    return jsonify(info)

if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        print("\n" + "="*60)
        print("🚀 IE7615 18014 - Group 5 Project Ready!")
        print(f"🌐 Open in browser: http://localhost:5000")
        print("-" * 60)
        print("📋 Configuration Summary:")
        print(f"   • Device: {DEVICE} ({'Fast' if DEVICE == 'cuda' else 'Slow - 1-2 min per image'})")
        print(f"   • Model: Fine-tuned on animal domain")
        print(f"   • Scheduler: Euler (best from evaluation)")
        print(f"   • CFG Scale: {BEST_CFG}")
        print(f"   • Steps: {NUM_STEPS}")
        print(f"   • Resolution: 512x512 (full quality)")
        print("-" * 60)
        if DEVICE == "cpu":
            print("⚠️  CPU Mode: Each image will take 1-3 minutes")
            print("💡 Tip: Use Google Colab for GPU acceleration")
        else:
            print("⚡ GPU Mode: Fast generation (~10-15 seconds)")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(debug=False, host='127.0.0.1', port=5000)
    else:
        print("\n❌ Failed to load model. Please check:")
        print("   1. Model path is correct")
        print("   2. All model files are present")
        print("   3. You have enough memory")
        print("   4. Try reinstalling diffusers: pip install --upgrade diffusers")