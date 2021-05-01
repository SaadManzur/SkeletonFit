import numpy as np

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyrr import Matrix44

from core.skeleton import Skeleton, EDGES, LEFTS, RIGHTS
from core.h36m_dataset import H36MDataset

curr_frame = 0
cylinder_off = True

dataset = H36MDataset("/home/saad/Personal/Research/Dataset/H36M/h36m-fetch", 
                      "/home/saad/Personal/Research/Experiments/MartinezBaseline/data/h36m/cameras.h5")

pose3d, cam = dataset.load_file("S1", "Directions-1", 54138969)
R, T, f, c, k, p, cam_id = cam

print(cam_id)

T = T.reshape((1, 3))
up = R.T[:, 1]
look = R.T[:, 2]

R_m44 = Matrix44.from_matrix33(R)

pygame.init()
display = (1000, 1000)
window = pygame.display.set_mode(display, DOUBLEBUF| OPENGL)

skeleton = Skeleton()

glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glShadeModel(GL_SMOOTH)
glEnable(GL_COLOR_MATERIAL)
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

sphere = gluNewQuadric()

glMatrixMode(GL_PROJECTION)
#49.64
gluPerspective(49.64, 1., 0.1, 10000.)

glMatrixMode(GL_MODELVIEW)
gluLookAt(T[0, 0], T[0, 1], T[0, 2], look[0], look[1], look[2], -up[0], -up[1], -up[2])
# glMultMatrixf(R_m44)
view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
print(view_matrix)
glLoadIdentity()

run = True

def draw_axis():

    glBegin(GL_LINES)
    glColor3f(1., 0., 0.)
    glVertex3f(0., 0., 0.)
    glVertex3f(1., 0., 0.)
    glEnd()
    
    glBegin(GL_LINES)
    glColor3f(0., 1., 0.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 1., 0.)
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0., 0., 1.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 0., 1.)
    glEnd()


def draw_skeleton(pose):

    limb_edges = []

    for (u, v) in EDGES:

        limb_edge = False
        glBegin(GL_LINES)
        
        if u in LEFTS and v in LEFTS:
            glColor3f(1., 0., 0.)
            limb_edge = True
        elif u in RIGHTS and v in RIGHTS:
            glColor3f(0., 0., 1.)
            limb_edge = True
        else:
            glColor3f(1., 1., 1.)

        if limb_edge:
            limb_edges.append((u, v))

        glVertex3f(pose[u, 0], pose[u, 1], pose[u, 2])
        glVertex3f(pose[v, 0], pose[v, 1], pose[v, 2])
        glEnd()

    if not cylinder_off:
        for (u, v) in limb_edges:

            u_to_v = pose[v] - pose[u]
            u_to_v_norm = np.linalg.norm(u_to_v)
            u_to_v_unit = u_to_v / u_to_v_norm
            angle = np.degrees(np.arccos(u_to_v_unit[2]))
            axis = np.cross(np.array([0., 0., 1.]), u_to_v_unit)
            u_v_mid = (pose[v] + pose[u]) / 2.

            cylinder = gluNewQuadric()

            glPushMatrix()
            height = u_to_v_norm*.85
            glColor4f(1., 1., 0., .45)
            glTranslatef(u_v_mid[0], u_v_mid[1], u_v_mid[2])
            glRotatef(angle, axis[0], axis[1], axis[2])
            glTranslatef(0., 0., -height/2.)
            gluCylinder(cylinder, 20., 20., height, 5, 10)
            glPopMatrix()

def draw():
    global curr_frame

    glLoadIdentity()
    glMultMatrixf(view_matrix)
    glPushMatrix()
    
    glLightfv(GL_LIGHT0, GL_POSITION, [1, -1, 1, 0])

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    draw_axis()

    draw_skeleton(pose3d[curr_frame, :, :])
    curr_frame = (curr_frame + 1) % pose3d.shape[0]

    # glColor4f(1.0, 0.0, 0.0, 1.0)
    # glTranslatef(-1.5, 0, 0)
    # gluSphere(sphere, 1.0, 32, 16)

    glPopMatrix()

while run:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                run = False
            elif event.key == pygame.K_c:
                cylinder_off = not cylinder_off
        
    draw()

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()