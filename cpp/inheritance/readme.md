# How It Works - Virtual Function Table:

## Each class gets its own vtable containing function pointer to all its functions

```
// When you create objects:
I2CDriver i2c_obj;
SPIDriver spi_obj;

// Each object has a vtable pointer:
CommDriver* ptr1 = &i2c_obj;   // ptr1 points to I2CDriver object
CommDriver* ptr2 = &spi_obj;   // ptr2 points to SPIDriver object

ptr1->send(...);  // Runtime lookup:
                  // 1. Follow vtable pointer in i2c_obj
                  // 2. Find send() function pointer  
                  // 3. Call I2CDriver::send()
Memory Layout:
i2c_obj:
├─ vtable_ptr → I2CDriver vtable
│               ├─ send() → I2CDriver::send
│               └─ status() → I2CDriver::status  
└─ [object data]

spi_obj:
├─ vtable_ptr → SPIDriver vtable  
│               ├─ send() → SPIDriver::send
│               └─ status() → SPIDriver::status
└─ [object data]
Real-World Usage:
cppvoid communicate_with_sensor(CommDriver* driver) {
    uint8_t cmd[] = {0x3B};  // Read command
    
    // Don't know/care which specific driver this is
    // Will call the correct derived implementation
    driver->send(cmd, 1);
    driver->status();
}

int main() {
    I2CDriver i2c;
    SPIDriver spi;
    
    communicate_with_sensor(&i2c);  // Calls I2C methods
    communicate_with_sensor(&spi);  // Calls SPI methods
    
    return 0;
}

```