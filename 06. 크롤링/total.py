from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import time

def save_이지스():
    url = 'https://www.incruit.com/company/1670741262'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인터크루_이지스.txt', 'w', encoding='utf-8') as file:
        example = soup.find('em')
        file.write(f'사업내용 : {example.text}\n\n')

        Overview = soup.find_all('div', {'class':'temp_grayBox'})
        file.write(f'회사 소개\n{Overview[0].text}\n\n')
        file.write('회사 연혁\n')
        for history in Overview[1]:
            file.write(f'{history.text}\n')

        aocnf = [soup.find_all('span', {'class' : 'gp_sty'})[i].text for i in [7, 4, 1]]
        file.write(f'\n연도별 매출[21년, 22년, 23년] : {aocnf}\n\n')

        driver = webdriver.Chrome()
        driver.get('https://www.egiskorea.com/company/directions.html')
        time.sleep(3)
        page = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        address = page.find('div', {'class':'group'}).find('p', {'class':'loca'}).contents[0].strip()
        file.write(f'주소 : {address}\n\n')

        company_info = soup.find('ul', {'myCompany_info'}).find_all('li')
        file.write(f'회사 {company_info[1].text}\n\n')

def save_더아이엠씨():
    url = 'https://www.incruit.com/company/1679124719'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인터크루_더아이엠씨.txt', 'w', encoding='utf-8') as file:
        example = soup.find('em')
        file.write(f'{example.text}\n\n')

        driver = webdriver.Chrome()
        driver.get('https://www.theimc.co.kr/web/php/sub/company_intro.php')
        page = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        intro = page.find_all('div', {'class':'ceo_desc'})
        file.write(f'회사 소개 : {intro[0].text[:37].strip()}\n\n')

        Overview = soup.find_all('div', {'class':'temp_grayBox'})
        file.write('회사연혁\n')
        for history in Overview[0]:
            file.write(f'{history.text}\n')

def save_이든티앤에스():
    url = 'https://www.incruit.com/company/25858679'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인크루트_이든티앤에스.txt', 'w', encoding='utf-8') as file:
        example = soup.find('em')
        file.write(f'사업내용 : {example.text}\n\n')

        Overview = soup.find_all('div', {'class':'temp_grayBox'})
        file.write(f'회사 소개\n{Overview[0].text}\n\n')

def save_엠엔비젼():
    url = 'https://www.incruit.com/company/1685023324'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인크루트_엠엔비젼.txt', 'w', encoding='utf-8') as file:
        example = soup.find('em')
        file.write(f'사업내용 : {example.text}\n\n')

def save_YH데이타베이스():
    url = 'https://www.incruit.com/company/1664785669'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인터크루_YH데이타베이스.txt', 'w', encoding='utf-8') as file:
        driver = webdriver.Chrome()
        driver.get('https://www.yhdatabase.com/portal/contents.do?mId=0201000000')
        time.sleep(3)
        page = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        company_profile = page.find('dl', {'class':'solution-content'})
        p = company_profile.find_all('p')
        file.write(f'회사 소개\n{p[1].text.strip()}\n\n')

def save_인터엑스():
    url = 'https://www.incruit.com/company/1688641863'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    with open('인터크루_인터엑스.txt', 'w', encoding='utf-8') as file:
        driver = webdriver.Chrome()
        driver.get('https://interxlab.com/about')
        time.sleep(3)
        page = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        dlsxj_x = page.find('div', {'class':'sub_about'})
        dlsxjdprtm = dlsxj_x.find_all('h2', {'class':'font_ns_400'})
        file.write(f'주요 사업 : {dlsxjdprtm[3].text}\n\n')

# 실행 루프
while True:
    print("\n1: 이지스\n2: 더아이엠씨\n3: 이든티앤에스\n4: 엠엔비젼\n5: YH데이타베이스\n6: 인터엑스\n0: 종료")
    choice = input("파일을 생성할 회사의 번호를 입력하세요: ")

    if choice == "0":
        print("프로그램을 종료합니다.")
        break
    elif choice == "1":
        save_이지스()
        print("이지스 파일이 생성되었습니다.")
    elif choice == "2":
        save_더아이엠씨()
        print("더아이엠씨 파일이 생성되었습니다.")
    elif choice == "3":
        save_이든티앤에스()
        print("이든티앤에스 파일이 생성되었습니다.")
    elif choice == "4":
        save_엠엔비젼()
        print("엠엔비젼 파일이 생성되었습니다.")
    elif choice == "5":
        save_YH데이타베이스()
        print("YH데이타베이스 파일이 생성되었습니다.")
    elif choice == "6":
        save_인터엑스()
        print("인터엑스 파일이 생성되었습니다.")
    else:
        print("올바른 번호를 입력하세요.")
