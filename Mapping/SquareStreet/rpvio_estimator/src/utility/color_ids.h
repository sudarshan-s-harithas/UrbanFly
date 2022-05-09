/**
 * This file implements the functionality for associting a plane id for a color in the mask image
 * Each plane id is assigned a color (r, g, b) as mentioned in seg_rgbs.txt
 * 
 **/
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

// Each color is stored in the hex format
std::map<unsigned long, int> color_index;

/**
 * Reads the color palette file at PALETTE_FILE_PATH
 * Assumes the following pattern in each line of the file 
 * (same as seg_rgbs.txt from airsim):
 * 
 * Currently expects seg_rgbs.txt from https://github.com/microsoft/AirSim/issues/3423#issuecomment-788077713
 * 
 * id [r, g, b]
 * (or)
 * id r g b
 * 
 * For example:
 * 0    [0, 0, 0]
 * 1    [12, 56, 134]
 * (or)
 * 0 0 0 0
 * 1 12 56 134
 **/
void load_color_palette(std::string PALETTE_FILE_PATH)
{
    std::ifstream palette_file(PALETTE_FILE_PATH);
    std::string row;

    // Read each line
    while(std::getline(palette_file, row))
    {
        std::stringstream row_stream(row);
	    std::vector<int> row_values;
        
        for (int value; row_stream >> value;) {
            row_values.push_back(value);
            
            char next = row_stream.peek();
            while ((next == ',' || next == ' ' || next == '[' || next == ']' || next == '\t') && !(next == '\n')) {
                row_stream.ignore();
                next = row_stream.peek();
            }
        }

        // Exatract r, g, b
        int id = row_values[0];
        int r = row_values[3];
        int g = row_values[2];
        int b = row_values[1];

        unsigned long hex = ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
        
        if ((r == 7) && (g == 93) && (b == 182)) {
            id = 0;    
        }

        if (color_index.find(hex) == color_index.end()) {
            color_index[hex] = id;
        }
    }

    std::cout << "********* Read color palette with " << std::to_string(color_index.size()) << " colors **********" << std::endl;
}

// Returns the id from color specified by (r, g, b) values
int color2id(int r, int g, int b)
{
    // TODO: Handle the ground plane in the mapping

    unsigned long hex = ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);

    if (color_index.find(hex) == color_index.end()) {
        color_index[hex] = 1000 + (int)color_index.size();
    }

    // std::cout << "Queried for color " << std::to_string(r) << " " << std::to_string(g) << " " << std::to_string(b) << std::endl;
    // std::cout << "ID is " << std::to_string(color_index[hex]) << std::endl;
    return color_index[hex];
}

// Returns the (r, g, b) values of a color in hex format
unsigned long id2color(int id)
{
    for (auto it = color_index.begin(); it != color_index.end(); ++it) {
        if (it->second == id)
            return it->first;   
    }
}