import random
import sys

import pygame
import numpy as np

from Button import Button
from incremental.delaunay2D import Delaunay2D

SCREEN_HEIGHT = 768
SCREEN_WIDTH = 1280
# Displaying all points and edges
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
clock = pygame.time.Clock()
zoom = 2


def mesh_inc(surface, numSeeds):
    radius = 730
    seeds = radius * np.random.random((numSeeds, 2)) + 5
    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * radius)
    print("Adding points...")
    print(seeds)
    for s in seeds:
        dt.addPoint(s)
        pygame.draw.circle(surface, "#CC00CC", s, 5)
    asd = dt.generateVoronoi()
    for t in dt.exportTriangles()[0]:
        print(t)
        pygame.draw.polygon(surface=surface, color="Green", points=[seeds[t[0]], seeds[t[1]], seeds[t[2]]], width=1)


def add_inc(surface):
    surface.fill((160, 160, 160))
    s = (random.randint(5, 735), random.randint(5, 735))
    seeds.append(s)
    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * radius)
    print(seeds)
    for s in seeds:
        dt.addPoint(s)
        pygame.draw.circle(surface, "#CC00CC", s, 5)

    for t in dt.exportTriangles()[0]:
        print(t)
        pygame.draw.polygon(surface=surface, color="Green", points=[seeds[t[0]], seeds[t[1]], seeds[t[2]]], width=1)


def clear(surface):
    surface.fill((160, 160, 160))
    global seeds
    seeds = []

def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/Roboto-Regular.ttf", size)


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect)


NOP_Info = get_font(16).render("No of Points:", True, "Black")
NOP_RECT = pygame.Rect(1150, 260, 140, 32)
radius = 730
center = (380, 380)
seeds = []


def inc_screen():
    pygame.display.set_caption("Incremental Algorithm")

    fake_screen = screen.copy()
    pic = pygame.surface.Surface((760, 760))
    pic.fill((160, 160, 160))
    zoom_size = (round(760 / zoom), round(760 / zoom))

    no_of_points = ""
    active = False
    zoomActive = False

    while True:
        mouse = pygame.mouse.get_pos()
        fake_screen.fill((160, 160, 160))
        fake_screen.blit(pic, (1, 1))
        screen.blit(pygame.transform.scale(fake_screen, screen.get_rect().size), (0, 0))
        if zoomActive:
            zoom_area = pygame.Rect(0, 0, *zoom_size)
            zoom_area.center = (mouse[0], mouse[1])
            zoom_surf = pygame.Surface(zoom_area.size)
            zoom_surf.blit(screen, (0, 0), zoom_area)
            zoom_surf = pygame.transform.scale(zoom_surf, (760, 760))
            screen.blit(zoom_surf, (0, 0))

        NOP_TEXT = get_font(16).render(no_of_points, True, "Black")
        pygame.draw.rect(screen, "Black", NOP_RECT, 2)
        screen.blit(NOP_Info, (NOP_RECT.x - 225, NOP_RECT.y + 5))
        screen.blit(NOP_TEXT, (NOP_RECT.x + 5, NOP_RECT.y + 5))
        NOP_RECT.w = max(100, NOP_TEXT.get_width() + 10)

        MESH = Button(image=None, pos=(1020, 340), text_input="MESH",
                      font=get_font(30), base_color="White", hovering_color="Green")
        STEP = Button(image=None, pos=(1080, 420), text_input="STEP-BY-STEP",
                      font=get_font(30), base_color="White", hovering_color="Green")
        CLEAR = Button(image=None, pos=(1190, 340), text_input="CLEAR",
                      font=get_font(30), base_color="White", hovering_color="Green")
        BACK = Button(image=None, pos=(1175, 700), text_input="BACK",
                      font=get_font(30), base_color="White", hovering_color="Green")

        for button in [MESH, STEP, CLEAR, BACK]:
            button.changeColor(mouse)
            button.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if BACK.checkForInput(mouse):
                    main_menu()
                elif MESH.checkForInput(mouse):
                    no = int(no_of_points)
                    mesh_inc(pic, no)
                elif CLEAR.checkForInput(mouse):
                    clear(pic)
                elif STEP.checkForInput(mouse):
                    add_inc(pic)
                if NOP_RECT.collidepoint(event.pos):
                    active = True
                else:
                    active = False
                if mouse <= (760, 760):
                    zoomActive = not zoomActive

            elif event.type == pygame.KEYDOWN and active:
                if event.key == pygame.K_BACKSPACE:
                    no_of_points = no_of_points[:-1]
                else:
                    no_of_points += event.unicode

        pygame.display.flip()
        clock.tick(90)


def main_menu():
    pygame.display.set_caption("Delaunay Triangulation Main Menu")
    screen.fill((160, 160, 160))
    while True:
        mouse = pygame.mouse.get_pos()

        menu_text = get_font(100).render("Main Menu", True, "#CC00CC")
        menu_rect = menu_text.get_rect(center=(640, 100))
        screen.blit(menu_text, menu_rect)

        INC_BUTTON = Button(image=None, pos=(640, 350), text_input="Incremental Algorithm",
                            font=get_font(40), base_color="White", hovering_color="Green")

        for button in [INC_BUTTON]:
            button.changeColor(mouse)
            button.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if INC_BUTTON.checkForInput(mouse):
                    inc_screen()

        pygame.display.update()


main_menu()