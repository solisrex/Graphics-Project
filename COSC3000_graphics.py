import glfw
import OpenGL
from OpenGL.GL import *
from ctypes import *
import numpy as np
from PIL import Image
from glm import perspective, ortho, radians, value_ptr, translate, rotate, identity, mat4, vec3

vShaderSource = """
#version 330 core
 layout (location = 0) in vec3 vPos;
 layout (location = 1) in vec2 cardTexCoord;

 uniform mat4 proj_matrix;
 uniform float time;
 uniform mat4 model;
 out vec2 fCardTexCoord;

 mat3 localRotation = mat3(cos(time),0,-sin(time),
                    0,1,0,
                    sin(time),0,cos(time));

 mat4 globalRotation = mat4(cos(time/10),0,-sin(time/10),0,
                    0,1,0,0,
                    sin(time/10),0,cos(time/10),0,
                    0,0,0,1);

 void main(void) {
    gl_Position = proj_matrix*globalRotation*model*vec4(localRotation*vPos+vec3(0,0,20),1.0f);
    fCardTexCoord = cardTexCoord;
}
"""


fShaderSource = """
#version 330 core
in vec2 fCardTexCoord;

out vec4 color;
uniform sampler2D back;
uniform sampler2D front;
uniform sampler2D suit;
uniform sampler2D value;
void main(void) {
    if (gl_FrontFacing) {
        vec4 frontColor = texture(front,fCardTexCoord);
        vec4 suitColor = texture(suit,fCardTexCoord);
        float suitAlpha = suitColor[3];
        vec4 valueColor = texture(value,fCardTexCoord);
        float valueAlpha = valueColor[3];
        color = (1-max(valueAlpha,suitAlpha))*frontColor+suitAlpha*suitColor+valueAlpha*valueColor;
    } else {
        color = texture(back, fCardTexCoord);
    }
}
"""

vertices = np.array([
    -0.9,-1.2,0,
    0.9,-1.2,0,
    0.9,1.2,0,
    -0.9,1.2,0
],np.float32)
cardTexCoords = np.array([
    1.0,0.0,
    0.0,0.0,
    0.0,1.0,
    1.0,1.0
],np.float32)


indices = np.array([
    0,1,3,
    1,2,3
],np.uint)


def main(backDesign,frontDesign,heartsDesign,spadesDesign,diamondsDesign,clubsDesign,redFaces,blackFaces):
    def window_size_callback(window, width, height):
        windowWidth = width
        windowHeight = height
        projMatrix = perspective(radians(30.0),windowWidth/windowHeight,0.1,100)
        glUniformMatrix4fv(render_projection_matrix_loc, 1, GL_FALSE, value_ptr(projMatrix))
        glViewport(0, 0, width, height)
    # Initialize the library
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    windowWidth = 1000
    windowHeight = 750
    window = glfw.create_window(windowWidth, windowHeight, "Durak!", None, None)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_window_size_limits(window, 500, 500, glfw.DONT_CARE,glfw.DONT_CARE)
    if not window:
        glfw.terminate()
        return
    
    # Make the window's context current
    glfw.make_context_current(window)
    glViewport(0, 0, windowWidth, windowHeight)

    # https://github.com/tito/pymt/blob/master/pymt/graphx/shader.py

    vShader = glCreateShader(GL_VERTEX_SHADER)
    fShader = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vShader,[vShaderSource])
    glShaderSource(fShader,[fShaderSource])

    glCompileShader(vShader)
    glCompileShader(fShader)
    
    program = glCreateProgram()
    glAttachShader(program, vShader)
    glAttachShader(program, fShader)
    glLinkProgram(program)

    success = glGetShaderiv(vShader,GL_COMPILE_STATUS)
    print("vshader:",success)
    success = glGetShaderiv(fShader,GL_COMPILE_STATUS)
    print("fshader:",success)
    success = glGetProgramiv(program,GL_LINK_STATUS)
    print("program:",success)

    vao = glGenVertexArrays(52)
    vbo = glGenBuffers(52)
    ebo = glGenBuffers(1)

    for i in range(52):
        suit = i // 13
        face = i % 13
        glBindVertexArray(vao[i])
        glBindBuffer(GL_ARRAY_BUFFER,vbo[i])
        glBufferData(GL_ARRAY_BUFFER,vertices.nbytes+cardTexCoords.nbytes,None,GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER,0,vertices.nbytes,vertices)
        glBufferSubData(GL_ARRAY_BUFFER,vertices.nbytes,cardTexCoords.nbytes,cardTexCoords)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,0,ctypes.c_void_p(vertices.nbytes))
        glEnableVertexArrayAttrib(vao[i],0)
        glEnableVertexArrayAttrib(vao[i],1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,4*len(indices),indices,GL_STATIC_DRAW)
        
    backTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,backTexture)
    backImg = Image.open(backDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,backImg.width,backImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,backImg.tobytes())
    backImg.close()
    glGenerateMipmap(GL_TEXTURE_2D)

    frontTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,frontTexture)
    frontImg = Image.open(frontDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,frontImg.width,frontImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,frontImg.tobytes())
    frontImg.close()
    glGenerateMipmap(GL_TEXTURE_2D)

    heartsTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,heartsTexture)
    heartsImg = Image.open(heartsDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,heartsImg.width,heartsImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,heartsImg.tobytes())
    heartsImg.close()
    glGenerateMipmap(GL_TEXTURE_2D)

    spadesTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,spadesTexture)
    spadesImg = Image.open(spadesDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,spadesImg.width,spadesImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,spadesImg.tobytes())
    spadesImg.close()
    glGenerateMipmap(GL_TEXTURE_2D)

    diamondsTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,diamondsTexture)
    diamondsImg = Image.open(diamondsDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,diamondsImg.width,diamondsImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,diamondsImg.tobytes())
    diamondsImg.close()
    glGenerateMipmap(GL_TEXTURE_2D)
    
    clubsTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,clubsTexture)
    clubsImg = Image.open(clubsDesign)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,clubsImg.width,clubsImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,clubsImg.tobytes())
    clubsImg.close()
    glGenerateMipmap(GL_TEXTURE_2D) # Need this for some reason. Investigate!

    redValueTextures = glGenTextures(13)
    blackValueTextures = glGenTextures(13)
    for i in range(13):
        glBindTexture(GL_TEXTURE_2D,redValueTextures[i])
        redFaceImg = Image.open(redFaces[i])
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,redFaceImg.width,redFaceImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,redFaceImg.tobytes())
        redFaceImg.close()
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D,blackValueTextures[i])
        blackFaceImg = Image.open(blackFaces[i])
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,blackFaceImg.width,blackFaceImg.height,0,GL_RGBA,GL_UNSIGNED_BYTE,blackFaceImg.tobytes())
        blackFaceImg.close()
        glGenerateMipmap(GL_TEXTURE_2D)

    # https://stackoverflow.com/questions/19993078/looking-for-a-simple-opengl-3-2-python-example-that-uses-glfw


    glUseProgram(program)

    render_projection_matrix_loc = glGetUniformLocation(program, "proj_matrix")
    time_loc = glGetUniformLocation(program, "time")
    tex0 = glGetUniformLocation(program, "back")
    tex1 = glGetUniformLocation(program, "front")
    tex2 = glGetUniformLocation(program, "suit")
    tex3 = glGetUniformLocation(program, "value")
    glUniform1i(tex0,0)
    glUniform1i(tex1,1)
    glUniform1i(tex2,2)
    glUniform1i(tex3,3)

    model_loc = glGetUniformLocation(program, "model")


    projMatrix = perspective(radians(30),windowWidth/windowHeight,0.1,100)
    # https://github.com/Zuzu-Typ/PyGLM/issues/1
    glUniformMatrix4fv(render_projection_matrix_loc, 1, GL_FALSE,value_ptr(projMatrix))

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here, e.g. using pyOpenGL
        glUniform1f(time_loc,glfw.get_time())
        glClearColor(0.1,0.2,0.4,1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, backTexture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, frontTexture)

        for i in range(52):
            suit = i // 13
            color = suit % 2
            value = i % 13
            model_matrix = rotate(np.pi*i/26,vec3(0,1,0))
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, value_ptr(model_matrix))
            if value == 0:
                glActiveTexture(GL_TEXTURE2)
                glBindTexture(GL_TEXTURE_2D, [heartsTexture,spadesTexture,diamondsTexture,clubsTexture][suit])
            if color == 0:
                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_2D, redValueTextures[value])
            else:
                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_2D, blackValueTextures[value])

            glBindVertexArray(vao[i])
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)

            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,None) # This blasted line of code wouldn't work because I used 0 instead of None! Freaking unbelieveable my dude!

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()



if __name__ == "__main__":
    backDesign = "./deckPics/backDesign.png"
    frontDesign = "./deckPics/frontDesign.png"
    heartsDesign = "./suitPics/red/hearts.png"
    spadesDesign = "./suitPics/black/spades.png"
    diamondsDesign = "./suitPics/red/diamonds.png"
    clubsDesign = "./suitPics/black/clubs.png"
    redFaces = ["./faceValue/red/red{0}.png".format(i) for i in range(2,11)]+["./faceValue/red/redJack.png","./faceValue/red/redQueen.png","./faceValue/red/redKing.png","./faceValue/red/redAce.png"]
    blackFaces = ["./faceValue/black/black{0}.png".format(i) for i in range(2,11)]+["./faceValue/black/blackJack.png","./faceValue/black/blackQueen.png","./faceValue/black/blackKing.png","./faceValue/black/blackAce.png"]
    main(backDesign,frontDesign,heartsDesign,spadesDesign,diamondsDesign,clubsDesign,redFaces,blackFaces)
    print("Goodbye!")

# Kessenich, John; Sellers, Graham; Shreiner, Dave. OpenGL Programming Guide (p. 512). Pearson Education. Kindle Edition. 
# https://stackoverflow.com/questions/19993078/looking-for-a-simple-opengl-3-2-python-example-that-uses-glfw
# https://stackoverflow.com/questions/74408404/sending-a-pointer-to-glvertexattributepointer-in-pyrthon
# https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/4.1.textures/textures.cpp
# https://github.com/tito/pymt/blob/master/pymt/graphx/shader.py
# https://learnopengl.com/Getting-started/Coordinate-Systems
# https://github.com/Zuzu-Typ/PyGLM/issues/1