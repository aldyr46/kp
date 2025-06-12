
import joblib

# Load model .pkl
model = joblib.load("model_kepribadian.pkl")

# Mapping label model ke nama karakter
label_map = {0: "Koleris", 1: "Melankolis", 2: "Plegmatis", 3: "Sanguinis"}

# Mapping karakter ke divisi kerja + alasan
rekomendasi_map = {
    "Koleris": {
        "divisi": "Produksi",
        "alasan": (
            "Anda cenderung mengambil keputusan cepat dan berorientasi pada target, "
            "yang sangat berguna di lini produksi. Apabila Anda juga tertarik pada analisis angka, "
            "Divisi Keuangan dapat menjadi pilihan alternatif."
        )
    },
    "Melankolis": {
        "divisi": "Keuangan",
        "alasan": (
            "Anda teliti dan perfeksionis, cocok untuk laporan keuangan, audit, atau administrasi yang terstruktur."
        )
    },
    "Plegmatis": {
        "divisi": "HRD",
        "alasan": (
            "Anda pendengar yang baik dan penengah konflik, cocok menangani urusan karyawan di HRD."
        )
    },
    "Sanguinis": {
        "divisi": "Pemasaran",
        "alasan": (
            "Anda komunikatif dan mudah bergaul, sangat cocok untuk menjalin hubungan dengan pihak eksternal."
        )
    }
}

def prediksi_kepribadian(jawaban):
    """
    Menerima list 12 jawaban (skala 1–5), mengembalikan hasil prediksi kepribadian dan divisi kerja.
    
    Parameters:
    jawaban (list[int]): 12 angka dari kuesioner
    
    Returns:
    dict: karakter, divisi, dan alasan rekomendasi
    """
    if len(jawaban) != 12:
        return {"error": "Jumlah jawaban harus 12"}
    
    pred = model.predict([jawaban])[0]
    karakter = label_map[pred]
    rekom = rekomendasi_map[karakter]
    
    return {
        "karakter": karakter,
        "divisi": rekom["divisi"],
        "alasan": rekom["alasan"]
    }

# Contoh penggunaan untuk testing mandiri
if __name__ == "__main__":
    jawaban = [4, 5, 4, 3, 2, 2, 1, 1, 1, 3, 3, 2]
    hasil = prediksi_kepribadian(jawaban)
    print("Karakter:", hasil["karakter"])
    print("Divisi Rekomendasi:", hasil["divisi"])
    print("Alasan:", hasil["alasan"])
