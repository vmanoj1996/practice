class Person
{
    // declare here. cannot init/allocate here since it can result in conflicts when there are multiple files inheriting this header
    public: static int population;

    public:
    Person()
    {
        population++;
    }

    static int getPopulation()
    {
       return population; 
    }

    int getPopulationInstance()
    {
        return population;
    }
};


extern int bla[];