// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include <GL/glew.h>
#include <SFML/Window.hpp>

// Shader sources
const GLchar* vertexSource = R"glsl(
    #version 150 core
    in vec2 position;
    in vec3 color;
    out vec3 Color;
    void main()
    {
        Color = color;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)glsl";
const GLchar* fragmentSource = R"glsl(
    #version 150 core
    in vec3 Color;
    out vec4 outColor;
    void main()
    {
        outColor = vec4(Color, 1.0);
    }
)glsl";


// void Keyboard(unsigned char key, int x, int y)
// {
//   switch (key)
//   {
//   case 27:             // ESCAPE key
// 	  exit (0);
// 	  break;
//   case 'l':
// 	  SelectFromMenu(MENU_LIGHTING);
// 	  break;
//   case 'p':
// 	  SelectFromMenu(MENU_POLYMODE);
// 	  break;
//   case 't':
// 	  SelectFromMenu(MENU_TEXTURING);
// 	  break;
//   }
// }
// int BuildPopupMenu (void)
// {
//   int menu;
//   menu = glutCreateMenu (SelectFromMenu);
//   glutAddMenuEntry ("Toggle lighting\tl", MENU_LIGHTING);
//   glutAddMenuEntry ("Toggle polygon fill\tp", MENU_POLYMODE);
//   glutAddMenuEntry ("Toggle texturing\tt", MENU_TEXTURING);
//   glutAddMenuEntry ("Exit demo\tEsc", MENU_EXIT);
//   return menu;
// }

// int main(int argc, char** argv)
// {
//   // GLUT Window Initialization:
//   glutInit (&argc, argv);
//   glutInitWindowSize (g_Width, g_Height);
//   glutInitDisplayMode ( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
//   glutCreateWindow ("CS248 GLUT example");
//   // Initialize OpenGL graphics state
//   InitGraphics();
//   // Register callbacks:
//   glutDisplayFunc (display);
//   glutReshapeFunc (reshape);
//   glutKeyboardFunc (Keyboard);
//   glutMouseFunc (MouseButton);
//   glutMotionFunc (MouseMotion);
//   glutIdleFunc (AnimateScene);
//   // Create our popup menu
//   BuildPopupMenu ();
//   glutAttachMenu (GLUT_RIGHT_BUTTON);
//   // Get the initial time, for use by animation
// #ifdef _WIN32
//   last_idle_time = GetTickCount();
// #else
//   gettimeofday (&last_idle_time, NULL);
// #endif
//   // Turn the flow of control over to GLUT
//   glutMainLoop ();
//   return 0;
// }

int test()
{
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;

    sf::Window window(sf::VideoMode(800, 600, 32), "OpenGL", sf::Style::Titlebar | sf::Style::Close, settings);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    // Create Vertex Array Object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create a Vertex Buffer Object and copy the vertex data to it
    GLuint vbo;
    glGenBuffers(1, &vbo);

    GLfloat vertices[] = {
        -0.5f,  0.5f, 1.0f, 0.0f, 0.0f, // Top-left
         0.5f,  0.5f, 0.0f, 1.0f, 0.0f, // Top-right
         0.5f, -0.5f, 0.0f, 0.0f, 1.0f, // Bottom-right
        -0.5f, -0.5f, 1.0f, 1.0f, 1.0f  // Bottom-left
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Create an element array
    GLuint ebo;
    glGenBuffers(1, &ebo);

    GLuint elements[] = {
        0, 1, 2,
        2, 3, 0
    };

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    // Create and compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Specify the layout of the vertex data
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), 0);

    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));

    bool running = true;
    while (running)
    {
        sf::Event windowEvent;
        while (window.pollEvent(windowEvent))
        {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                running = false;
                break;
            }
        }

        // Clear the screen to black
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw a rectangle from the 2 triangles using 6 indices
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Swap buffers
        window.display();
    }

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);

    glDeleteVertexArrays(1, &vao);

    window.close();

    return 0;
}
