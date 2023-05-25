import sys
import pygame
import numpy as np
import time
import math

from Button import Button
from incremental.voronooi2Dincremental import Voronoi2DIncremental

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)


def get_font(size):
    return pygame.font.Font("assets/Karla-Regular.ttf", size)


TESTING = 1
SCREEN_HEIGHT = 768
SCREEN_WIDTH = 1280
pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
clock = pygame.time.Clock()
zoom = 2
spacing = 10
NOP_Info = get_font(30).render("N: ", True, "Black")
NOP_RECT = pygame.Rect(1150, 310, 140, 30)
radius = SCREEN_HEIGHT / 2
center = ((SCREEN_HEIGHT + 250) / 2, SCREEN_HEIGHT / 2)
points = []

def generate_random_points(numPoints):
    global center
    global points
    random_numbers_btw_zero_and_two_pi = np.random.random(numPoints)*2*math.pi
    random_numbers_btw_zero_and_radius_min_spacing = np.random.random(size=numPoints)
    points.clear()
    for i in range(numPoints):
        theta__ = clamp(random_numbers_btw_zero_and_two_pi[i], 0, 2*math.pi)
        radius__ = clamp(math.sqrt(random_numbers_btw_zero_and_radius_min_spacing[i])*(radius - spacing), 0, radius - spacing)
        x_coord = radius__*math.cos(theta__) + center[0]
        y_coord = radius__*math.sin(theta__) + center[1]
        points.append(np.array([x_coord, y_coord]))


def generate_randomized_incremental(surface, numSeeds):
    global center
    global radius
    generate_random_points(numSeeds)
    clock_start = time.time()
    dt = Voronoi2DIncremental(center, 50 * radius)
    for p in points:
        dt.addPoint(p)
    voronoi_edges, voronoi_vertices = dt.generateVoronoi()
    clock_end = time.time()
    for t in dt.exportTriangles()[0]:
        pygame.draw.polygon(surface=surface, color=(181, 230, 29), points=[points[t[0]], points[t[1]], points[t[2]]], width=1)
    for v_e in voronoi_edges:
        pygame.draw.line(surface=surface, color="#CC00FF", start_pos=v_e[0], end_pos=v_e[1], width=2)    
    for v_v in voronoi_vertices:
        pygame.draw.circle(surface, "#CCCC11", v_v, 1)
    for p in points:
        pygame.draw.circle(surface, "#FF0000", p, 2)
    if TESTING:
        print("The Voronoi diagram with %d points took %f ms: " % (numSeeds, (clock_end - clock_start)*1000.0))


def generate_flipping(surface, numSeeds):
    global center
    global radius
    generate_random_points(numSeeds)
    clock_start = time.time()
    dt = Voronoi2DIncremental(center, 50 * radius)
    for p in points:
        dt.addPoint(p)
    voronoi_edges, voronoi_vertices = dt.generateVoronoi()
    clock_end = time.time()
    for t in dt.exportTriangles()[0]:
        pygame.draw.polygon(surface=surface, color=(181, 230, 29), points=[points[t[0]], points[t[1]], points[t[2]]], width=1)
    for v_e in voronoi_edges:
        pygame.draw.line(surface=surface, color="#CC00FF", start_pos=v_e[0], end_pos=v_e[1], width=2)    
    for v_v in voronoi_vertices:
        pygame.draw.circle(surface, "#CCCC11", v_v, 1)
    for p in points:
        pygame.draw.circle(surface, "#FF0000", p, 2)
    if TESTING:
        print("The Voronoi diagram with %d points took %f ms: " % (numSeeds, (clock_end - clock_start)*1000.0))


def clear(surface):
    surface.fill((76, 98, 122))
    global points
    points = []


def rand_inc_event_loop():
    pygame.display.set_caption("Randomized Incremental Algorithm")

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
        CLEAR = Button(image=None, pos=(1148, 420), text_input="Clear All",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")
        BACK = Button(image=None, pos=(1150, 725), text_input="Back to Main Menu",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")

        for button in [GENERATE, CLEAR, BACK]:
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
                    if no_of_points == "":
                        no_of_points = "10"
                    no = int(no_of_points)
                    consecutive_nexts = 0
                    clear(pic)
                    generate_randomized_incremental(pic, no)
                elif CLEAR.checkForInput(mouse):
                    consecutive_nexts = 0
                    clear(pic)
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
        clock.tick()


def flipping_event_loop():
    pygame.display.set_caption("Flipping Algorithm")

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
        CLEAR = Button(image=None, pos=(1148, 420), text_input="Clear All",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")
        BACK = Button(image=None, pos=(1150, 725), text_input="Back to Main Menu",
                      font=get_font(25), base_color="Black", hovering_color="Yellow")

        for button in [GENERATE, CLEAR, BACK]:
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
                    if no_of_points == "":
                        no_of_points = "10"
                    no = int(no_of_points)
                    consecutive_nexts = 0
                    clear(pic)
                    generate_randomized_incremental(pic, no)
                elif CLEAR.checkForInput(mouse):
                    consecutive_nexts = 0
                    clear(pic)
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
        clock.tick()
    return

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
                    rand_inc_event_loop()
                elif FLIP_BUTTON.checkForInput(mouse):
                    flipping_event_loop()
                elif QUIT.checkForInput(mouse):
                    pygame.quit()
                    sys.exit()
                    
        pygame.display.flip()
        clock.tick()
        

main_menu()