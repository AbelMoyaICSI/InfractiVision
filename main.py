import tkinter as tk
from src.app_manager import AppManager

def main():
    root = tk.Tk()
    root.title("InfractiVision - Principal")
    root.geometry("1280x720")
    root.state("zoomed") 
    app = AppManager(root)
    app.show_welcome()    
    root.mainloop()

if __name__ == "__main__":
    main()
