from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

url = "https://www.jailbreakchat.com/"

driver = webdriver.Chrome()

driver.get(url)

sleep(1)

elems = driver.find_elements(By.CSS_SELECTOR,'p')

with open("jailbreak_prompts.csv", "w") as f:
    f.write(",act,prompt\n")
    for e in elems:
        text = e.text.replace("\n", "").strip()
        text = e.text.replace('"', '""')
        f.write(f',,\"{text}"\n')
