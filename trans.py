import pandas as pd

# Fungsi untuk mengkategorikan angka berdasarkan rentang
def categorize(number):
    if 1 <= number <= 14:
        return 0
    elif 15 <= number <= 24:
        return 1
    elif 25 <= number <= 34:
        return 2
    elif 35 <= number <= 44:
        return 3
    elif 45 <= number <= 54:
        return 4
    elif 55 <= number <= 64:
        return 5
    elif 65 <= number <= 75:
        return 6
    elif 75 < number:
        return 7
    # Tambahkan rentang lain sesuai kebutuhan

def glucose_cat(number):
    if number < 180.0 :
        return 0
    elif number >= 180.0 :
        return 1

def bmi_cat(number):
    if number < 30.0 :
        return 0
    elif number == 30.0 :
        return 1
    elif number > 30.0 :
        return 2

# Baca data dari file Excel
# file_path = 'brain_stroke2.csv'  # Ganti dengan path ke file Excel Anda
# df = pd.read_excel(file_path)
df = pd.read_csv('brain_stroke2.csv')

# Asumsi kolom data Anda bernama 'Angka', ganti sesuai nama kolom yang sebenarnya
df['age'] = df['age'].apply(categorize)
df['avg_glucose_level'] = df['avg_glucose_level'].apply(glucose_cat)
df['bmi'] = df['bmi'].apply(bmi_cat)

# Simpan hasilnya ke file CSV baru
output_file_path = 'new_brain_stroke.csv'  # Ganti dengan path ke file output yang diinginkan
df.to_csv(output_file_path, index=False)

print("Data berhasil dikategorikan dan disimpan ke file CSV baru.")
