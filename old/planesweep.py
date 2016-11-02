import numpy as np
from PIL import Image
from glumpy import app, gloo, gl, data, glm
from load_camera import CameraLoader, CameraRigType, camera_0

width = 1024/2
height = 768/2
ratio = float(width)/height

window = app.Window(width=width, height=height)

camera = CameraLoader()

posx = -10.0
angleR = 0.0

def cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3)]
    itype = np.uint32

    # Vertices positions
    h = height
    w = width
    d = 0
    p = np.array([
        (-1, 1, d),
        (1, 1, d),
        (1, -1, d),
        (-1, -1, d),
        ], dtype=float)

    print points

    # Face Normals
    n = np.array([[0, 0, 1], [0, 0, 1]])

    # Texture coords
    t = np.array([
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1]
    ])

    faces_p = [0, 1, 2, 3]

    faces_n = [0, 0, 0, 0]
    faces_t = [0, 1, 2, 3]

    vertices = np.zeros(4, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal'] = n[faces_n]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)

    return vertices, filled


def normalCam(w, h):
    fovy = 3.0
    aspect = w / float(h)
    zNear = 1.0
    zFar = 100.0
    return glm.perspective(fovy, aspect, zNear, zFar)


@window.event
def on_resize(width, height):
    print "ON RESIZE"
    #program['u_projection'] = glm.perspective(fovy, aspect, zNear, zFar)
    program['u_projection'] = normalCam(1024, 768)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)

@window.event
def on_draw(dt):

    window.clear()
    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    program.draw(gl.GL_TRIANGLES, indices)

    #model = np.eye(4, dtype=np.float32)
    #program['u_model'] = model
    #program['u_model'] = glm.xrotate(glm.translation(0, 0, posx), angleR)
    #program['u_model'] = np.eye(4)
    program['u_model'] = glm.zrotate(np.eye(4),angleR).dot(glm.translation(0, 0, posx))


@window.event
def on_key_press(symbol, modifiers):
    global posx, angleR

    # Press key 1
    if symbol == 49:
        print "[LEFT] Camera"
        program['u_projection'] = camera.get_camera(CameraRigType.Left)
    elif symbol == 50:
        print "[VIRTUAL] Camera"
        program['u_projection'] = camera.get_camera(CameraRigType.Virtual)
    elif symbol == 51:
        print "[RIGHT] Camera"
        program['u_projection'] = camera.get_camera(CameraRigType.Right)
    elif symbol == 32:
        print "Switch back to original camera"
        program['u_projection'] = normalCam(1024, 768)

    #UP 65362
    elif symbol == 65362:
        posx += 1.0
        print "Position X(up): %s" % posx
    #DOWN 65364
    elif symbol == 65364:
        posx -= 1.0
        print "Position X(down): %s" % posx

    #LEFT
    elif symbol == 65361:
        angleR += 1.0
        print "[ROTATE] X(left): %s" % angleR
    #RIGHT
    elif symbol == 65363:
        angleR -= 1.0
        print "[ROTATE] X(right): %s" % angleR

# Vertex Shader
vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix

attribute vec3 a_position;      // Vertex position
attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)

void main() {
    // Assign varying variables
    v_texcoord = a_texcoord;
    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

fragment = """
uniform sampler2D u_texture;  // Texture
varying vec2 v_texcoord;
void main() {
    vec4 t_color = texture2D(u_texture, v_texcoord);
    gl_FragColor = t_color;
}
"""


program = gloo.Program(vertex, fragment, count=4)
program['u_texture'] = np.asarray(Image.open('test_L.jpg'))
program['u_model'] = np.eye(4, dtype=np.float32)
#program['u_view'] = glm.translation(0, 0, -10.0)
program['u_view'] = np.eye(4, dtype=np.float32)

vertices, indices = camera_0.camera_plane_at()
program.bind(vertices)

app.run()