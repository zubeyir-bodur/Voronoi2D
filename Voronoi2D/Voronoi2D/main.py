import random
import sys

import pygame
import numpy as np

from Button import Button
from incremental.voronooi2Dincremental import Voronoi2DIncremental

SCREEN_HEIGHT = 768
SCREEN_WIDTH = 1280
# Displaying all points and edges
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
clock = pygame.time.Clock()
zoom = 2
spacing = 100

def mesh_inc(surface, numSeeds):
    radius = SCREEN_HEIGHT
    seeds = (radius + 250 - 2*spacing, radius  - 2*spacing) * np.random.random((numSeeds, 2)) + spacing
    center = np.mean(seeds, axis=0)
    dt = Voronoi2DIncremental(center, 50 * radius)
    print("Adding points...")
    print(seeds)
    for s in seeds:
        dt.addPoint(s)
        pygame.draw.circle(surface, "#FF0000", s, 7)
    voronoi = dt.generateVoronoi()
    for t in dt.exportTriangles()[0]:
        print(t)
        pygame.draw.polygon(surface=surface, color=(181, 230, 29), points=[seeds[t[0]], seeds[t[1]], seeds[t[2]]], width=2)
    for v_e in voronoi:
        print(v_e)
        pygame.draw.line(surface=surface, color="#CC00FF", start_pos=v_e[0], end_pos=v_e[1], width=2)


def add_inc(surface):
    surface.fill((76, 98, 122))
    s = (random.randint(spacing, SCREEN_HEIGHT + 250 - spacing), random.randint(spacing, SCREEN_HEIGHT-spacing))
    seeds.append(s)
    center = np.mean(seeds, axis=0)
    dt = Voronoi2DIncremental(center, 50 * radius)
    print(seeds)
    for s in seeds:
        dt.addPoint(s)
        pygame.draw.circle(surface, "#FF0000", s, 7)

    for t in dt.exportTriangles()[0]:
        print(t)
        pygame.draw.polygon(surface=surface, color=(181, 230, 29), points=[seeds[t[0]], seeds[t[1]], seeds[t[2]]], width=2)


def clear(surface):
    surface.fill((76, 98, 122))
    global seeds
    seeds = []

def get_font(size):
    return pygame.font.Font("assets/Karla-Regular.ttf", size)


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect)


NOP_Info = get_font(30).render("N: ", True, "Black")
NOP_RECT = pygame.Rect(1150, 310, 140, 30)
radius = 730
center = (380, 380)
seeds = []


def inc_screen():
    pygame.display.set_caption("Incremental Algorithm")

    fake_screen = screen.copy()
    pic = pygame.surface.Surface((SCREEN_HEIGHT + 250, SCREEN_HEIGHT))
    pic.fill((76, 98, 122))
    zoom_size = (round(SCREEN_HEIGHT / zoom), round(SCREEN_HEIGHT / zoom))

    no_of_points = ""
    active = False
    zoomActive = False
    consecutive_nexts = 0
    while True:
        mouse = pygame.mouse.get_pos()
        fake_screen.fill((161, 200, 207))
        fake_screen.blit(pic, (1, 1))
        screen.blit(pygame.transform.scale(fake_screen, screen.get_rect().size), (0, 0))
        if zoomActive:
            zoom_area = pygame.Rect(0, 0, *zoom_size)
            zoom_area.center = (mouse[0], mouse[1])
            zoom_surf = pygame.Surface(zoom_area.size)
            zoom_surf.blit(screen, (0, 0), zoom_area)
            zoom_surf = pygame.transform.scale(zoom_surf, (SCREEN_HEIGHT + 251, SCREEN_HEIGHT))
            screen.blit(zoom_surf, (0, 0))

        NOP_TEXT = get_font(25).render(no_of_points, True, "Black")
        pygame.draw.rect(screen, "Black", NOP_RECT, 2)
        screen.blit(NOP_Info, (NOP_RECT.x - 50, NOP_RECT.y - 5))
        screen.blit(NOP_TEXT, (NOP_RECT.x + 5, NOP_RECT.y))
        NOP_RECT.w = max(100, NOP_TEXT.get_width() + 10)

        GENERATE = Button(image=None, pos=(1150, 370), text_input="Generate",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")
        STEP = Button(image=None, pos=(1155, 420), text_input="Next Step",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")
        CLEAR = Button(image=None, pos=(1148, 470), text_input="Clear All",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")
        BACK = Button(image=None, pos=(1150, 725), text_input="Back to Main Menu",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")

        for button in [GENERATE, STEP, CLEAR, BACK]:
            button.changeColor(mouse)
            button.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if BACK.checkForInput(mouse):
                    consecutive_nexts = 0
                    main_menu()
                elif GENERATE.checkForInput(mouse):
                    no = int(no_of_points)
                    consecutive_nexts = 0
                    clear(pic)
                    mesh_inc(pic, no)
                elif CLEAR.checkForInput(mouse):
                    consecutive_nexts = 0
                    clear(pic)
                elif STEP.checkForInput(mouse):
                    if consecutive_nexts == 0:
                        clear(pic)
                    consecutive_nexts += 1
                    add_inc(pic)
                if NOP_RECT.collidepoint(event.pos):
                    active = True
                else:
                    active = False
                if mouse <= (SCREEN_HEIGHT + 251, SCREEN_HEIGHT):
                    zoomActive = not zoomActive

            elif event.type == pygame.KEYDOWN and active:
                if event.key == pygame.K_BACKSPACE:
                    no_of_points = no_of_points[:-1]
                else:
                    no_of_points += event.unicode

        pygame.display.flip()
        clock.tick(90)


def main_menu():
    pygame.display.set_caption("CS 478 Project - Implementing Three Voronoi Diagram Computation Algorithms " 
                               + "and Comparing Their Performance")
    screen.fill((161, 200, 207))
    while True:
        mouse = pygame.mouse.get_pos()

        menu_text = get_font(30).render("Implementation of Voronoi Diagram", True, (126, 47, 40))
        menu_rect = menu_text.get_rect(center=(275, 100))
        screen.blit(menu_text, menu_rect)
        menu_text = get_font(30).render("Algorithms in Python", True, (126, 47, 40))
        menu_rect = menu_text.get_rect(center=(275, 130))
        screen.blit(menu_text, menu_rect)

        INC_BUTTON = Button(image=None, pos=(200, 250), text_input="Randomized Incremental",
                            font=get_font(25), base_color="Black", hovering_color="Yellow")

        FORTUNE_BUTTON = Button(image=None, pos=(165, 300), text_input="Fortune's Algorithm",
                            font=get_font(25), base_color="Black", hovering_color="Yellow")
        FLIP_BUTTON = Button(image=None, pos=(160, 350), text_input="Flipping Algorithm",
                            font=get_font(25), base_color="Black", hovering_color="Yellow")
        QUIT = Button(image=None, pos=(75, 400), text_input="Exit",
                            font=get_font(25), base_color="Black", hovering_color="Yellow")

        for button in [INC_BUTTON, FORTUNE_BUTTON, FLIP_BUTTON, QUIT]:
            button.changeColor(mouse)
            button.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if INC_BUTTON.checkForInput(mouse):
                    inc_screen()
                elif QUIT.checkForInput(mouse):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()


main_menu()