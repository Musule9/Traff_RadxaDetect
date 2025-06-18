import cv2
import time
import sys

def test_oscar_rtsp():
    # Tu URL RTSP específica
    rtsp_url = "rtsp://admin:OscarWilde2016.@10.0.30.53:554/VideoInput/1/h264/1"
    
    print("🔗 Probando RTSP de Oscar...")
    print(f"URL: {rtsp_url}")
    print()
    
    # Conectar
    print("📡 Conectando...")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_TIMEOUT, 10000)  # 10 segundos timeout
    
    if not cap.isOpened():
        print("❌ FALLO: No se pudo conectar al RTSP")
        print()
        print("🔧 Posibles problemas:")
        print("   1. IP 10.0.30.53 no accesible")
        print("   2. Puerto 554 bloqueado")
        print("   3. Credenciales incorrectas")
        print("   4. Cámara apagada o sin red")
        return False
    
    print("✅ CONECTADO al RTSP!")
    
    # Obtener info del stream
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Información del stream:")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS configurado: {fps}")
    print()
    
    # Test de lectura de frames
    print("📹 Probando lectura de frames...")
    frames_ok = 0
    frames_error = 0
    start_time = time.time()
    
    try:
        for i in range(150):  # ~5 segundos a 30fps
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frames_ok += 1
                
                # Mostrar progreso cada 30 frames
                if i % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frames_ok / elapsed if elapsed > 0 else 0
                    print(f"   Frame {frames_ok:3d} | FPS actual: {current_fps:5.1f} | Resolución real: {frame.shape[1]}x{frame.shape[0]}")
            else:
                frames_error += 1
                
            # Si hay muchos errores consecutivos, parar
            if frames_error > 20:
                print("❌ Demasiados errores consecutivos")
                break
                
    except Exception as e:
        print(f"❌ Error durante lectura: {e}")
    
    finally:
        cap.release()
    
    # Resultados
    total_time = time.time() - start_time
    fps_promedio = frames_ok / total_time if total_time > 0 else 0
    tasa_exito = (frames_ok / (frames_ok + frames_error)) * 100 if (frames_ok + frames_error) > 0 else 0
    
    print()
    print("📊 RESULTADOS FINALES:")
    print(f"   ✅ Frames exitosos: {frames_ok}")
    print(f"   ❌ Frames con error: {frames_error}")
    print(f"   ⏱️ Tiempo total: {total_time:.1f} segundos")
    print(f"   📈 FPS promedio: {fps_promedio:.1f}")
    print(f"   📊 Tasa de éxito: {tasa_exito:.1f}%")
    print()
    
    # Evaluación
    if frames_ok >= 100 and tasa_exito >= 80:
        print("🎉 RESULTADO: ✅ STREAM RTSP COMPLETAMENTE FUNCIONAL")
        print("   📋 El stream de tu cámara está trabajando perfectamente.")
        print("   📋 Si el sistema principal falla, el problema es en el procesamiento.")
        return True
    elif frames_ok > 20 and tasa_exito >= 50:
        print("⚠️ RESULTADO: 🟡 STREAM RTSP FUNCIONAL PERO INESTABLE")
        print("   📋 El stream funciona pero con algunos problemas.")
        print("   📋 Revisa la calidad de la conexión de red.")
        return True
    else:
        print("❌ RESULTADO: 🔴 STREAM RTSP NO FUNCIONAL")
        print("   📋 Hay problemas serios con la conectividad.")
        return False

if __name__ == "__main__":
    success = test_oscar_rtsp()
    sys.exit(0 if success else 1)