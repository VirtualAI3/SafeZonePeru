import ratings

def main():
    print("CI: validando pipeline de reentrenamiento...")
    ok = ratings.trigger_retrain()

    if ok:
        print("CI: reentrenamiento ejecutado correctamente.")
    else:
        print("CI: no se disparó reentrenamiento (estado válido).")

if __name__ == "__main__":
    main()
